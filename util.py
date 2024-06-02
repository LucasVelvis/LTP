from typing import List, Type
from models.model import Model
from models.falcon import Falcon
from models.zephyr import Zephyr
from prompting_techniques.prompt import Prompt
from prompting_techniques.zero_shot import ZeroShot
from prompting_techniques.few_shot import FewShot
from prompting_techniques.automatic_cot import AutomaticCoT
from prompting_techniques.generated_knowledge import GeneratedKnowledge


def get_all_models() -> List[Type[Model]]:
    return [Falcon, Zephyr]


def get_all_prompting_techniques() -> List[Type[Prompt]]:
    return [ZeroShot, FewShot, AutomaticCoT, GeneratedKnowledge]
