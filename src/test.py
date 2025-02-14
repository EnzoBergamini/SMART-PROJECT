import os
import pytest


class TestTextFiles:
    @pytest.mark.parametrize("directory_path", ["../datasets/data/labels/test"])
    def test_lines_have_five_words(self, directory_path):
        """Test to verify if the annotated labels have a good format. (5 values per line)

        Args:
            path to the repertory with the annoted label text files.

        Returns:
            None
        """
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)

                with open(file_path, "r") as file:
                    lines = file.readlines()

                    for line_number, line in enumerate(lines, 1):
                        words = line.split()
                        assert len(words) == 5, (
                            f"File '{filename}', Line {line_number}: Expected 5 words, found {len(words)} words"
                        )

    @pytest.mark.parametrize("directory_path", ["../datasets/data/labels/test"])
    def test_first_word_is_integer_between_0_and_10(self, directory_path):
        """Test to verify if the annotated labels exist. (value between 0 and 9)

        Args:
            path to the repertory with the annoted label text files.

        Returns:
            None
        """
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)

                with open(file_path, "r") as file:
                    lines = file.readlines()

                    for line_number, line in enumerate(lines, 1):
                        words = line.split()
                        if words:
                            first_word = words[0]

                            try:
                                first_word_int = int(first_word)
                                assert 0 <= first_word_int <= 9, (
                                    f"File '{filename}', Line {line_number}: First word '{first_word}' is not between 0 and 9"
                                )
                            except ValueError:
                                assert False, (
                                    f"File '{filename}', Line {line_number}: First word '{first_word}' is not an integer"
                                )
