from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from torch.nn.functional import softmax
from enum import Enum
from typing import Optional


class Language(Enum):
    ZH = 'zh'
    EN = 'en'

class Model:
    _instance: 'Model' = None
    _model: AutoModelForCausalLM = None
    _tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = None
    _device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _PROMPTS: dict[Language, list[str]] = {
    Language.ZH: [
        """* "禁止危險內容": 提示不得包含或尋求生成對自己和/或他人造成傷害的內容（例如：獲取或製造槍械和爆炸裝置、宣傳恐怖主義、教唆自殺的指示）。""",
        """* "禁止騷擾": 提示不得包含或尋求生成針對他人的惡意、恐嚇、霸凌或辱罵性的內容（例如：人身威脅、否認悲劇事件、貶低暴力受害者）。""",
        """* "禁止仇恨言論": 提示不得包含或尋求生成基於種族、性別、族裔、宗教、國籍、性取向、殘疾狀況或種姓的仇恨言論，亦不得煽動或宣傳此類仇恨。""",
        """* "禁止露骨性資訊": 提示不得包含或尋求生成涉及性行為或其他猥褻內容的資訊（例如：色情描述、旨在引起性興奮的內容）。與人體解剖或性教育相關的醫學或科學術語則被允許。"""
    ],

    Language.EN: [
        """* "No Dangerous Content": The prompt shall not contain or seek generation of content that harming oneself and/or others (e.g., accessing or building firearms and explosive devices, promotion of terrorism, instructions for suicide).""",
        """* "No Harassment": The prompt shall not contain or seek generation of content that is malicious, intimidating, bullying, or abusive content targeting another individual (e.g., physical threats, denial of tragic events, disparaging victims of violence).""",
        """* "No Hate Speech": The prompt shall not contain or seek generation of content that targeting identity and/or protected attributes (e.g., racial slurs, promotion of discrimination, calls to violence against protected groups) or dehumanizing/belittling/vilifying on the bases of race, ethnicity, religion, disability, age, nationality, veteran status, sexual orientation, gender, gender identity, caste, or any other protected status. We do allow content that includes a reference to a protected individual or group in a positive manner.""",
        """* "No Sexually Explicit Information": The prompt shall not contain or seek generation of content that contains references to sexual acts or other lewd content (e.g., sexually graphic descriptions, content aimed at causing arousal). Medical or scientific terms related to human anatomy or sex education are permitted."""
    ]
}

    @staticmethod
    def get_instance():
        if (Model._instance is None):
            Model()
        return Model._instance

    def __init__(self):
        if Model._instance is not None:
            raise Exception('only one instance can exist')
        else:
            self._id = id(self)
            Model._instance = self
            Model._instance._initalize()

    def _initalize(self):
        model_name = "google/shieldgemma-2b"
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir="cache")
        self._tokenizer.padding_side = "right"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, load_in_4bit=True, cache_dir="cache")
        # self._model.to(dtype = torch.bfloat16)
        # if (torch.cuda.is_available()):
        #     self._model.cuda()

    def run(self, input: str, language: Language = Language.EN):
        model_input = self._tokenizer(
            self._add_context(input, language),
            return_tensors="pt",
            padding="longest"
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**model_input).logits
        vocab = self._tokenizer.get_vocab()
        selected_logit = logits[:, -1, [vocab['Yes'], vocab['No']]]
        softmax_logit = softmax(selected_logit, dim=1)
        probabilities = [logit[0].item() for logit in softmax_logit]
        return [softmax_logit, probabilities]

    def _add_context(self, input: str, language: Language) -> list[str]:
        outputs : list[str] = []
        for prompt in self._PROMPTS[language]:
            output = "You are a policy expert trying to help determine whether a user prompt is in violation of the defined safety policies.\n"
            output += "\n"
            output += "<start_of_turn>\n"
            output += f"Human Question: {input.strip()}\n"
            output += "<end_of_turn>\n"
            output += "\n"
            output += "Our safety principle is defined in the below:"
            output += "\n"
            output += prompt.strip()
            output += "\n"
            output += "Does the human question violate the above principle? Your answer must start with 'Yes' or 'No'. And then walk through step by step to be sure we answer correctly."
            outputs.append(output)
        return outputs