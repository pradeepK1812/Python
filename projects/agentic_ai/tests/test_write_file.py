from agentic_ai.tools import write_file, read_file


def test_write_and_read_file():
    test_file = "test_output.txt"
    expected_content = "Hello from our Agentic AI tool!"

    write_file(test_file, expected_content)

    content = read_file(test_file)

    assert content == expected_content
