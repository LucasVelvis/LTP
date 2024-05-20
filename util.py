from models.falcon import Falcon
from prompting_techniques.zero_shot import ZeroShot
from prompting_techniques.few_shot import FewShot
from typing import List
from models.model import Model
from prompting_techniques.prompt import Prompt

def get_all_models() -> List[Model]:
    return [Falcon]

def get_all_prompting_techniques() -> List[Prompt]:
    return [ZeroShot, FewShot]