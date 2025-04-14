import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
from src.utils import load_config, setup_logger, get_output_writer, initialize_capture
from src.detector import MotionDetector

def main():
    # Đọc cấu hình
    config = load_config("config/settings.yaml")
    logger = setup_logger(config['log']['log_dir'])

    # Khởi tạo video capture
    try:
        cap, source_info = initialize_capture(config)
    except ValueError as e:
        logger.error(str(e))
        print(str(e))
        return

    logger.info("Đã mở nguồn thành công: %s", source_info)
    print(f"Đã mở nguồn thành công: {source_info}")
    frame_width = config['video']['frame_width']
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * frame_width / cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Khởi tạo detector
    detector = MotionDetector(config)
    out = None
    motion_detected = False

    # Đo FPS
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Hết video hoặc lỗi đọc frame")
            print("Hết video hoặc lỗi đọc frame")
            break

        # Lật khung hình nếu dùng camera
        if config['video']['source'] == 'camera':
            frame = cv2.flip(frame, 1)

        # Xử lý khung hình
        processed_frame, text, motion_detected_now, object_count = detector.process_frame(frame, frame_width)

        # Ghi video và lưu ảnh khi có chuyển động
        if motion_detected_now and out is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out, output_path = get_output_writer(config, frame_height, timestamp)
            logger.info("Bắt đầu ghi video: %s", output_path)
            print(f"Bắt đầu ghi video: {output_path}")
            motion_detected = True

        if motion_detected_now:
            snapshot_path = os.path.join(config['video']['output_dir'], f"snapshot_{timestamp}.jpg")
            cv2.imwrite(snapshot_path, processed_frame)
            logger.info("Lưu ảnh chụp: %s", snapshot_path)
            print(f"Lưu ảnh chụp: {snapshot_path}")

        if out is not None:
            out.write(processed_frame)

        if text == "Normal" and motion_detected:
            motion_detected = False
            if out is not None:
                out.release()
                logger.info("Đã dừng ghi video")
                print("Đã dừng ghi video")
                out = None

        # Hiển thị khung hình
        cv2.imshow("Surveillance Feed", processed_frame)

        # Đo FPS
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            logger.info("FPS: %.2f", fps_actual)
            print(f"FPS: {fps_actual:.2f}")

        # Xử lý phím
        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # Esc
            logger.info("Thoát bởi người dùng")
            print("Thoát bởi người dùng")
            break
        elif key == ord('r'):
            detector.reset_background()
            logger.info("Reset nền")

    # Giải phóng tài nguyên
    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Đã giải phóng tài nguyên")
    print("Đã giải phóng tài nguyên")

if __name__ == "__main__":
    main()
