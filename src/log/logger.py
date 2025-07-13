# write_log.py

from datetime import datetime

class Logger:
    """
    로그를 파일에 저장하고,
    필요 시 콘솔에도 출력하는 Logger 클래스
    """
    def __init__(self, log_path: str, print_also: bool = True):
        self.log_path = log_path
        self.print_also = print_also

    def write(self, message: str, print_also: bool = True, print_error: bool = False):
        """
        로그 메시지를 파일에 기록하고,
        print_also=True일 경우 콘솔에도 출력
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"{timestamp} | {message}\n"
        # 로그 파일에 append 모드로 기록
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)
            
        # 옵션에 따라 콘솔 출력
        if self.print_also:
            if print_error:
                print(f"\033[91m{line}\033[0m", end='')
            else:
                print(line, end='')
