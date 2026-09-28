from rag.retrieve import main as ask


def main(argv=None) -> int:
    return ask(argv, trace=True)


if __name__ == "__main__":
    raise SystemExit(main())
