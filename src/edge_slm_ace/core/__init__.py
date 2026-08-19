"""ACE roles: the prompts the Reflector and Curator run on.

The Generator is gone: under loglikelihood option scoring the model never
generates an answer, so there is no generator prompt and no answer to parse.
Only reflection and curation still need a model to write text.
"""

from edge_slm_ace.core.ace_roles import (
    build_curator_prompt,
    choose_lessons_for_playbook,
    parse_curator_output,
    parse_reflector_output_to_lessons,
)

__all__ = [
    "build_curator_prompt",
    "choose_lessons_for_playbook",
    "parse_curator_output",
    "parse_reflector_output_to_lessons",
]
