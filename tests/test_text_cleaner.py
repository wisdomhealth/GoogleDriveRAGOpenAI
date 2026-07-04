from app.utils.text_cleaner import clean_text


def test_clean_text_normalizes_whitespace_and_hyphenation() -> None:
    raw = "This is a hyphen-\nated word.\r\n\r\n\r\n  Extra   spaces  "

    assert clean_text(raw) == "This is a hyphenated word.\n\nExtra spaces"
