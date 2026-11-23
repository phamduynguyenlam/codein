import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test as base_tester


def main():
    base_tester.main(agent_name="disc_af")


if __name__ == "__main__":
    main()
