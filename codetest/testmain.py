import fibo
import sys
# getattr(sys.modules[__name__], "Foo")

if __name__ == "__main__":
    try:
        a = 10 / 0
    finally:
        print(1)