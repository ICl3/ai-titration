from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import threading
import json
import os
import time
import cv2
import experiment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'titration-lab'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
exp = experiment.TitrationExperiment()
exp_thread = None
idle_thread = None


def load_configs():
    path = os.path.join(BASE_DIR, 'experiment_configs.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _log_to_client(level, message):
    socketio.emit('log_message', {'level': level, 'message': message})


exp.on_log = _log_to_client


# ---- 路由 ----

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/app')
def app_page():
    return render_template('app.html')


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with exp.frame_lock:
                if exp.latest_frame is not None:
                    ret, buf = cv2.imencode('.jpg', exp.latest_frame,
                                            [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_bytes = buf.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n'
                               + frame_bytes + b'\r\n')
            time.sleep(0.1)

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/configs')
def api_configs():
    return jsonify(load_configs())


@app.route('/api/status')
def api_status():
    return jsonify({
        'running': exp.running,
        'finished': exp.finished,
        'volume': exp.total_volume,
        'color': exp.current_color,
        'confidence': exp.current_confidence,
        'pump_connected': exp.pump_ser is not None,
        'ph_connected': exp.ph_ser is not None,
        'camera_connected': exp.camera is not None,
    })


@app.route('/api/cameras')
def api_cameras():
    return jsonify(exp.list_cameras())


# ---- SocketIO 事件 ----

@socketio.on('connect_hardware')
def handle_connect_hardware(data=None):
    results = {}

    camera_idx = data.get('camera_idx') if data else None

    try:
        exp.connect_camera(camera_idx)
        results['camera'] = exp.camera is not None
    except Exception as e:
        results['camera'] = False
        _log_to_client('error', f'摄像头连接失败: {e}')

    try:
        exp.connect_pump()
        results['pump'] = exp.pump_ser is not None
    except Exception as e:
        results['pump'] = False
        _log_to_client('error', f'泵连接失败: {e}')

    try:
        exp.connect_ph_meter()
        results['ph_meter'] = exp.ph_ser is not None
    except Exception as e:
        results['ph_meter'] = False
        _log_to_client('error', f'pH计连接失败: {e}')

    results['simulation_mode'] = (exp.pump_ser is None)
    emit('hardware_status', results)

    if exp.camera is not None:
        start_idle_capture()


@socketio.on('start_experiment')
def handle_start_experiment(data):
    global exp_thread, idle_thread

    if exp.running:
        emit('log_message', {'level': 'error', 'message': '实验已在运行中'})
        return

    config_id = data.get('config_id', 'methyl_orange')
    configs = load_configs()
    if config_id not in configs:
        emit('log_message', {'level': 'error', 'message': f'未知实验类型: {config_id}'})
        return

    config = configs[config_id]

    try:
        if exp.camera is None:
            exp.connect_camera()
            if exp.camera is None:
                emit('log_message', {'level': 'error', 'message': '摄像头不可用，无法启动'})
                return

        # 自动连接泵（如果尚未连接）
        if exp.pump_ser is None:
            exp.connect_pump()

        if exp.pump_ser is None:
            emit('log_message', {'level': 'warning',
                 'message': '泵未连接，将以模拟模式运行（AI识别正常但不实际滴液）'})

        detection_mode = data.get('detection_mode', config.get('detection_mode', 'dl'))
        exp.detection_mode = detection_mode

        exp.load_model(config['weights_path'], config['class_json_path'])
        exp.endpoint_color = config['endpoint_color']
        exp.pump_speed_rpm = data.get('speed_rpm', config['titrate_speed_rpm'])
        exp.confidence_threshold = data.get('confidence', config['confidence_threshold'])
        exp.cycle_interval = data.get('cycle_interval', config.get('cycle_interval_sec', 0.5))
        exp.double_confirm = data.get('double_confirm', config.get('double_confirm', True))
        exp.hsv_confidence_threshold = data.get('hsv_confidence_threshold', config.get('hsv_confidence_threshold', 0.35))
        exp.endpoint_streak_frames = data.get('endpoint_streak_frames', config.get('endpoint_streak_frames', 2))
        exp.hybrid_penalty = data.get('hybrid_penalty', config.get('hybrid_penalty', 0.7))

        exp.on_update = lambda d: socketio.emit('experiment_update', d)

        hsv_config = {
            'hsv_target': config.get('hsv_target', {}),
            'roi_size': config.get('roi_size', 200),
            'endpoint_color': config['endpoint_color'],
        }
        exp_thread = threading.Thread(
            target=exp.run_experiment_loop,
            args=(detection_mode, hsv_config),
            daemon=True
        )
        exp_thread.start()

        emit('log_message', {'level': 'info', 'message': f'实验已启动: {config["name"]} (模式: {detection_mode})'})

    except Exception as e:
        emit('log_message', {'level': 'error', 'message': f'启动失败: {e}'})


@socketio.on('stop_experiment')
def handle_stop_experiment():
    exp.stop()
    emit('experiment_stopped', {'message': '实验已停止，可开始新实验'})


@socketio.on('purge_bubbles')
def handle_purge_bubbles(data):
    if exp.running:
        emit('purge_status', {'ok': False, 'message': '实验进行中，请先停止实验再排气泡'})
        return

    if exp.pump_ser is None:
        exp.connect_pump()

    if exp.pump_ser is None:
        emit('purge_status', {'ok': False, 'status': 'done',
              'message': '蠕动泵未连接，无法排气泡。请检查泵接线和电源'})
        return

    direction = data.get('direction', 'ccw')
    duration_sec = data.get('duration_sec', 10)
    if duration_sec < 1:
        duration_sec = 10
    if duration_sec > 60:
        duration_sec = 60

    def on_start(total, dir_name):
        socketio.emit('purge_status', {
            'ok': True, 'status': 'running', 'total': total, 'direction': dir_name
        })

    def on_tick(remaining):
        socketio.emit('purge_status', {
            'ok': True, 'status': 'running', 'remaining': remaining
        })

    def on_done(ok, message):
        socketio.emit('purge_status', {
            'ok': ok, 'status': 'done', 'message': message
        })

    exp.purge_bubbles(
        direction, duration_sec,
        on_start=on_start, on_tick=on_tick, on_done=on_done
    )


@socketio.on('shutdown_system')
def handle_shutdown(data=None):
    _log_to_client('info', '正在关闭系统...')
    exp.shutdown()
    socketio.emit('shutdown_complete')
    time.sleep(0.5)
    os._exit(0)


# ---- 空闲摄像头读取线程 ----

def start_idle_capture():
    global idle_thread
    if idle_thread is not None and idle_thread.is_alive():
        return
    idle_thread = threading.Thread(target=exp.idle_capture_loop, daemon=True)
    idle_thread.start()


# ---- 启动 ----

if __name__ == '__main__':
    import webbrowser

    def open_browser():
        time.sleep(1.5)
        webbrowser.open('http://localhost:5000')

    threading.Thread(target=open_browser, daemon=True).start()
    print("服务器启动中...")
    print("浏览器将自动打开 http://localhost:5000")
    print("按 Ctrl+C 停止服务器")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
