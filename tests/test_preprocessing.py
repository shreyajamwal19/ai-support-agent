import sys
sys.path.insert(0, ".")
from src.data.reconstruct import clean_text_preserve, content_hash


def test_url_stripped():
    assert "[URL]" in clean_text_preserve("check this https://example.com/abc out")


def test_mention_stripped():
    assert "@AmazonHelp" not in clean_text_preserve("@AmazonHelp help me")


def test_whitespace_normalized():
    assert clean_text_preserve("a    b\n\nc") == "a b c"


def test_content_hash_deterministic():
    assert content_hash("Hello World") == content_hash("hello world  ")


def test_content_hash_differs_for_different_text():
    assert content_hash("abc") != content_hash("xyz")


def test_empty_and_none_handled():
    assert clean_text_preserve("") == ""
    assert clean_text_preserve(None) == ""
