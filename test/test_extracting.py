import re

def test_extracting(content):
    pattern = r"```python\s*\n(.*?)\n```"
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


if __name__ == "__main__":
    with open("test/test_extracting_md.md", "r", encoding="utf-8") as file:
        content = file.read()
    print(test_extracting(content=content))
