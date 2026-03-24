from pathlib import Path
from typing import Generic, Optional, TypeVar

T = TypeVar("T")

PROMPT_PATH = Path(__file__).parent.parent.parent.parent.joinpath("conf", "prompts")


def _load_prompt(prompt_path: Path, section: Optional[str] = None, **kwargs) -> str:
    """
    Load prompt content from a markdown file.
    Supports variable substitution using {variable_name} syntax.
    Can extract specific sections by header name.

    Args:
        filepath: Path to the markdown file
        section: Optional section header to extract (without ##)
        **kwargs: Variables to inject into the prompt

    Returns:
        Full content or specific section content with variables injected

    Examples:
        # Load entire file
        load_prompt(Path("prompt.md"), name="John")

        # Load specific section
        load_prompt(Path("prompt.md"), section="Analysis Instructions", name="John")
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract specific section if requested
    if section:
        content = _extract_section(content, section)

    # Inject variables
    if kwargs:
        content = content.format(**kwargs)

    return content


def _extract_section(content: str, section_name: str) -> str:
    """
    Extract a specific section from markdown content by header name.
    Supports ## headers at any level.

    Args:
        content: Full markdown content
        section_name: Header name to find (without ## prefix)

    Returns:
        Content of the section (excluding the header itself)

    Raises:
        ValueError: If section is not found
    """
    lines = content.split("\n")
    section_lines = []
    in_section = False
    section_level = None

    for line in lines:
        # Check if this is a header line
        if line.strip().startswith("#"):
            # Parse header level and title
            header_match = line.strip().lstrip("#")
            header_level = len(line.strip()) - len(header_match)
            header_title = header_match.strip()

            # Check if this is our target section
            if header_title.lower() == section_name.lower():
                in_section = True
                section_level = header_level
                continue  # Skip the header itself

            # If we're in a section and hit a same/higher level header, we're done
            elif in_section and header_level <= section_level:
                break

        # Collect lines if we're in the target section
        if in_section:
            section_lines.append(line)

    if not section_lines:
        raise ValueError(f"Section '{section_name}' not found in markdown file")

    return "\n".join(section_lines).strip()


class Prompt(Generic[T]):
    _prompt_path: Path

    def __init__(self):
        self._current_prompt: str = "Main prompt not formatted yet"

        with open(self._prompt_path, "r") as file:
            self._prompt = file.read()

    def _update(self, **opts) -> str:
        """
        Format the original-prompt with the given formatting updates
        """
        self._current_prompt = self._prompt.format_map(opts)
        return self._current_prompt


class DatasetDescription(Prompt):
    _prompt_path = PROMPT_PATH.joinpath("dataset_description.md")

    def update(
        self,
        dataset_name: str,
        num_rows: int,
        num_columns: int,
        dataset_metadata: dict,
        column_details: dict,
        sample_data,
    ) -> str:
        return self._update(
            **{
                "dataset_name": dataset_name,
                "num_rows": num_rows,
                "num_columns": num_columns,
                "dataset_metadata": dataset_metadata,
                "column_details": column_details,
                "sample_data": sample_data,
            }
        )


class CandidatesDiscoveryPrompt(Prompt):
    _prompt_path = PROMPT_PATH.joinpath("propose_discovery_tasks.md")

    def update(
        self,
        dataset_name: str,
        num_rows: int,
        num_columns: int,
        dataset_metadata: dict,
        column_details: dict,
        sample_data,
    ) -> str:
        query_dataset_prompt = DatasetDescription()
        query_dataset_description = query_dataset_prompt.update(
            dataset_name,
            num_rows,
            num_columns,
            dataset_metadata,
            column_details,
            sample_data,
        )

        return self._update(query_dataset_description=query_dataset_description)





class PandasStatementGenerationPrompt(Prompt):
    _prompt_path = PROMPT_PATH.joinpath("pandas_statement_generation.md")
    def __init__(self):
        super().__init__()
        self._datasets_descriptions = ""  

    def update(
        self,
        dataset_name: str,
        num_rows: int,
        num_columns: int,
        dataset_metadata: dict,
        column_details: dict,
        sample_data,
        aliases,
        matches
    ) -> str:
        query_dataset_prompt = DatasetDescription()
        query_dataset_description = query_dataset_prompt.update(
            dataset_name,
            num_rows,
            num_columns,
            dataset_metadata,
            column_details,
            sample_data,
        )
        self._datasets_descriptions += f"\n{query_dataset_description}"
        return self._update(table=self._datasets_descriptions,matches=matches,aliases=aliases)


class SQLStatementGenerationPrompt(Prompt):
    _prompt_path = PROMPT_PATH.joinpath("sql_statement_generation.md")
    def __init__(self):
        super().__init__()
        self._datasets_descriptions = ""  

    def update(
        self,
        dataset_name: str,
        num_rows: int,
        num_columns: int,
        dataset_metadata: dict,
        column_details: dict,
        sample_data,
        aliases,
        matches
    ) -> str:
        query_dataset_prompt = DatasetDescription()
        query_dataset_description = query_dataset_prompt.update(
            dataset_name,
            num_rows,
            num_columns,
            dataset_metadata,
            column_details,
            sample_data,
        )
        self._datasets_descriptions += f"\n{query_dataset_description}"
        return self._update(table=self._datasets_descriptions,matches=matches,aliases=aliases)
    


class JudgementResponseGenerationPrompt(Prompt):
    _prompt_path = PROMPT_PATH.joinpath("judge.md")
    def __init__(self):
        super().__init__()

    def update(
        self,
        data
    ) -> str:
        return self._update(data=data)
    

class ResponseGenerationPrompt(Prompt):
    _prompt_path = PROMPT_PATH.joinpath("response_generation.md")
    def __init__(self):
        super().__init__()

    def update(
        self,
        question,
        data
    ) -> str:
        return self._update(data=data,question=question)