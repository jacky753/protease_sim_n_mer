import time

from protease_rf import AppConfig
from protease_rf.app import build_and_run


def main() -> None:
    build_and_run(AppConfig())


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(elapsed)
    print(elapsed / 60)
    print(elapsed / 60 / 60)
    print(elapsed / 60 / 60 / 24)
    print("END.")
