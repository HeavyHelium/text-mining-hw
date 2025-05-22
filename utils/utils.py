import json
import time
import re
import pandas as pd
import matplotlib.pyplot as plt

from io import BytesIO
from functools import wraps
from typing import List, Tuple, Optional, Any
from PIL import Image
from ratelimit import limits, sleep_and_retry
from google import genai
from google.genai import types

MAX_CALLS_PER_MIN = 10


def retry(retries: int = 3, delay: float = 3.0):
    """
    Decorator to retry a function up to `retries` times with `delay` between attempts.
    Uses print statements to report retry attempts and failures. Returns None on final failure.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries:
                        print(f"Attempt {attempt} for {func.__name__} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"All {retries} attempts for {func.__name__} failed: {e}.")
            return None
        return wrapper
    return decorator


class VisionLanguageModel:
    """
    Wrapper for Google GenAI Vision-Language Model calls.
    Holds standard prompt templates as class variables.
    """

    PROMPT_TEMPLATE = (
        "You are a sophisticated Vision-Language Model (VLM) capable of analyzing images containing multiple-choice questions."
        " To guide your analysis, you may adopt the following process:\n"
        "0. Consider the subject of question is {subject} and image contains {content}.\n"
            "-{ocr}]n"
        "1. Image Analysis: Examine the image closely, identifying key elements such as text, diagrams, and any other relevant features.\n"
        "2. Question Text Extraction: Extract the text of the question\n"
        "3. Extract Answer Choices: Identify and extract the answer choices provided in the image \n"
        "    - if the answer options are not enumerated with letters, do enumerate them with letters (A, B, C, D, ...)\n"
        "4. Look for additional visual elements such as tables, diagrams, charts, or graphs.\n"
        "5. Ensure to consider any multilingual or multidomain aspects of the image, including text in different languages or mathematical/physics/scientific notation.\n"
        "6. Analyze the complete context and data provided\n"
        "7. Select correct answer based solely on analysis.\n"
        "8. Final answer by only the corresponding letter (single capital letter) without any extra explanation.\n"
            "- If the answer is not clear, still provide the best guess letter based on the analysis."
    )

    PROMPT_TEMPLATE_STRICTER = (
        "You are a sophisticated Vision-Language Model (VLM) capable of analyzing images containing multiple-choice questions."
        " To guide your analysis, you may adopt the following process:\n"
        "0. Consider the subject of question is {subject} and image contains {content}.\n"
        "1. Image Analysis: Examine the image closely, identifying key elements such as text, diagrams, and any other relevant features.\n"
            "-{ocr}]n"
        "2. Question Text Extraction: Extract the text of the question\n"
        "3. Extract Answer Choices: Identify and extract the answer choices provided in the image \n"
        "    - if the answer options are not enumerated with letters, do enumerate them with letters (A, B, C, D, ...)\n"
        "4. Look for additional visual elements such as tables, diagrams, charts, or graphs.\n"
        "5. Ensure to consider any multilingual or multidomain aspects of the image, including text in different languages or mathematical/physics/scientific notation.\n"
        "6. Analyze the complete context and data provided\n"
        "7. Select correct answer based solely on analysis.\n"
        "8. Respond by only the corresponding letter (single capital letter) without any extra explanation.\n"
        "9. If the answer is not clear, still provide the best guess as single capital letter.\n\n"
        "Always respond with a single capital letter (A, B, C, D, E) without any extra explanation."
    )

    PROMPT_TEMPLATE_BUL_STRICTER = (
        "Ти си комплексен Vision-Language модел (VLM) способен да анализира изображения, съдържащи multiple-choice questions."
        " В насочването на анализите си, подходи така:\n"
        "0. Вземи предвид, че предметът на въпроса е свързан с {subject} и изборажението съдържа {content}.\n"
            "-{ocr}]n"
        "1. Анализ на изображение: Изследвай отблизо изображението, идентифицирай ключови елементи като текст, диаграми, и всякакви други релевантни характеристики.\n"
        "2. Извлечи текстът, който представлява въпроса\n"
        "3. Идентифицирай и извлечи опциите за отговор на въпроса \n"
        "    - Ако отговорите не са номерирани с букви, номерирай ги с български букви (А, Б, В, Г, Д)\n"
        "4. Потърси допълнителни визуални елементи, като таблици, диаграми, графики или диаграми.\n"
        "5. Увери се, че вземаш предвид всички многоезични или многодоменни аспекти на изображението, включително текст на различни езици или математическа/физична/научна нотация.\n"
        "6. Анализирай целия контекст и предоставените данни\n"
        "7. Избери правилния отговор единствено въз основа на анализ.\n"
        "8. Отговори само със съответната буква (една главна буква) без допълнителни обяснения.\n"
        "9. Ако отговорът не е ясен, все пак посочи най-доброто предположение с една българска главна буква.\n\n"
        "Винаги отговаряй с една българска буква без никакви допълнителни обяснения."
    )

    PROMPT_TEMPLATE_BUL = (
        "Ти си комплексен Vision-Language модел (VLM) способен да анализира изображения, съдържащи multiple-choice questions."
        " В насочването на анализите си, подходи така:\n"
        "0. Вземи предвид, че предметът на въпроса е свързан с {subject} и изборажението съдържа {content}.\n"
            "-{ocr}]n"
        "1. Анализ на изображение: Изследвай отблизо изображението, идентифицирай ключови елементи като текст, диаграми, и всякакви други релевантни характеристики.\n"
        "2. Извлечи текстът, който представлява въпроса\n"
        "3. Идентифицирай и извлечи опциите за отговор на въпроса \n"
        "    - Ако отговорите не са номерирани с букви, номерирай ги с български букви (А, Б, В, Г, Д)\n"
        "4. Потърси допълнителни визуални елементи, като таблици, диаграми, графики или диаграми.\n"
        "5. Увери се, че вземаш предвид всички многоезични или многодоменни аспекти на изображението, включително текст на различни езици или математическа/физична/научна нотация.\n"
        "6. Анализирай целия контекст и предоставените данни\n"
        "7. Избери правилния отговор единствено въз основа на анализ.\n"
        "8. Отговори само със съответната буква (една главна буква) без допълнителни обяснения.\n"
        "9. Ако отговорът не е ясен, все пак посочи най-доброто предположение с една българска главна буква.\n\n"
    )


    PROMPT_CLASSIFICATION = (
        "Take following text and classify it into one of the following categories based on final answer:\n"
        "A, B, C, D, E\n"
        "Respond with only the letter of the classification and nothing else."
    )

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    @sleep_and_retry
    @limits(calls=MAX_CALLS_PER_MIN, period=60)
    def _generate(self,
                  model: str,
                  contents: List[Any],
                  thinking: bool = False, 
                  strip: bool = False) -> str:
        """
        Internal method to call the GenAI generate_content endpoint with rate limiting.
        """
        try:
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=1024)
            ) if thinking else None
            params = {} if config is None else {'config': config}
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                **params
            )
            return response.text.strip() if strip else response
        except Exception as e:
            print(f"Error in _generate: {e}")
            raise

    @retry(retries=3, delay=1.0)
    def get_answer(self,
                   prompt: str,
                   pil_image: Image.Image,
                   model: str,
                   strict: bool = False,
                   thinking: bool = True, 
                   strip=True) -> Optional[str]:
        """
        Send prepared prompt and PIL image to the VLM and return its response.
        """
        contents: List[Any] = []
        # if strict:
        #     #print("Using strict prompt for get_answer")
        #     contents.append("Always respond with a single capital letter (A-E) without explanation.")
        if pil_image is not None:
            contents.extend([prompt, pil_image])
        else:
            contents.append(prompt)
        #print(f"Calling model {model} with prompt length {len(prompt)}")
        return self._generate(model=model, contents=contents, thinking=thinking, strip=strip)

class DataUtils:
    """
    Utility methods for data loading, plotting, and JSON export.
    """

    @staticmethod
    def load_parquet(path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        print(f"Loaded DataFrame from {path}, shape={df.shape}")
        return df

    @staticmethod
    def plot_dataframe(df: pd.DataFrame, x: str, y: str) -> None:
        ax = df.plot(x=x, y=y)
        ax.set(xlabel=x, ylabel=y, title=f"{y} over {x}")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def save_to_json(data: Any, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved JSON to {filename}")


def visualize_image(image_bytes: bytes) -> None:
    img = Image.open(BytesIO(image_bytes))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def get_elements_contained(
        chem: bool,
        table: bool,
        figure: bool,
        graph: bool
    ) -> str:
    elems = []
    if chem: elems.append('chemical structure')
    if table: elems.append('table')
    if figure: elems.append('figure')
    if graph: elems.append('graph')
    return ', '.join(elems)


def prepare_content(
        row: pd.Series,
        strict: bool = False, 
        language: str = 'en',
        ocr_enrichment: bool = False,
    ) -> Tuple[str, Image.Image]:
    """
    Build prompt text and PIL Image from DataFrame row using class templates.
    """
    elements = get_elements_contained(
        bool(row.get('chemical_structure', False)),
        bool(row.get('table', False)),
        bool(row.get('figure', False)),
        bool(row.get('graph', False))
    )
    if language == 'en':
        template = VisionLanguageModel.PROMPT_TEMPLATE_STRICTER if strict else VisionLanguageModel.PROMPT_TEMPLATE
    elif language == 'bg':
        template = VisionLanguageModel.PROMPT_TEMPLATE_BUL_STRICTER if strict else VisionLanguageModel.PROMPT_TEMPLATE_BUL
    else:
        raise ValueError(f"Unsupported language: {language}")

    ocr_text = ""
    if ocr_enrichment:
        ocr = row['ocr_text']
        if ocr and language == 'bg':
            ocr_text = f"<begin-ocr>Вземи предвид OCR данните: {ocr}<end-ocr>\n"
        elif ocr and language == 'en':
            ocr_text = f"<begin-ocr>Consider OCR data: {ocr}<end-ocr>\n"

    prompt = template.format(subject=row['subject'], content=elements, ocr=ocr_text)
    #print(f"Prepared prompt starts with: {prompt[:50]}...")
    img = Image.open(BytesIO(row['image']['bytes']))
    return prompt, img


@retry(retries=3, delay=1.0)
def classify_answer(
        answer: str,
        client: genai.Client,
        model: str = "gemini-2.0-flash"
    ) -> Optional[str]:
    """
    Ensure answer is A-E; otherwise reclassify via LLM.
    """
    valid = {'A','B','C','D','E'}
    ans = answer.strip()
    if ans in valid:
        return ans
    prompt = VisionLanguageModel.PROMPT_CLASSIFICATION
    return client.models.generate_content(model=model, contents=[prompt, answer]).text.strip()



def get_answers(
        df: pd.DataFrame,
        vlm: VisionLanguageModel,
        model: str,
        strict: bool = False, 
        language: str = 'en',
        ocr_enrichment: bool = False,
        thinking: bool = True
    ) -> Tuple[List[dict], List[int]]:
    answers, failed = [], []
    #print(42)
    for i, row in df.iterrows():
        prompt, img = prepare_content(row, strict, language, ocr_enrichment)
        resp = vlm.get_answer(prompt, img, model, strict, thinking)
        # print(f"Answer for {row['sample_id']}: {resp}")
        # time.sleep(2)  # Rate limiting
        if resp is None:
            failed.append(i)
        else:
            answers.append({'id': row['sample_id'], 'answer_key': resp,
                            'language': row.get('language')})
    return answers, failed


def get_answers_chunked(
        df: pd.DataFrame,
        vlm: VisionLanguageModel,
        model: str,
        strict: bool = False,
        language: str = 'en',
        ocr_enrichment: bool = False,
        step: int = 10,
        thinking: bool = True
    ) -> Tuple[List[dict], List[int]]:
    all_ans, all_failed = [], []
    for start in range(0, len(df), step):
        end = min(start + step, len(df))
        a, f = get_answers(df.iloc[start:end], vlm, model, strict, language, ocr_enrichment, thinking)
        all_ans.extend(a)
        all_failed.extend(f)
    return all_ans, all_failed


def postprocess_answers(
        answers: List[dict],
        client: genai.Client,
        model: str = "gemini-2.0-flash"
    ) -> List[int]:
    failed = []
    for idx, ans in enumerate(answers):
        cls = classify_answer(ans['answer_key'], client, model)
        if cls is None:
            failed.append(idx)
        else:
            ans['answer_key'] = cls
    return failed


def make_gold_file_json(df: pd.DataFrame, path: str) -> None:
    data = [{'id': r['sample_id'], 'answer_key': r['answer_key'], 'language': r['language']}
            for _, r in df.iterrows()]
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Gold file saved to {path}")

def make_gold_file_json_bg(df: pd.DataFrame, path: str) -> None:
    data = [{'id': r['sample_id'], 'answer_key': r['answer_key'], 'language': r['language']}
            for _, r in df.iterrows()]
    
    for item in data:
        match item['answer_key']:
            case 'а' | 'А':
                item['answer_key'] = 'А'
            case 'б' | 'Б':
                item['answer_key'] = 'B'
            case 'в' | 'В':
                item['answer_key'] = 'C'
            case 'г' | 'Г':
                item['answer_key'] = 'D'
            case 'д' | 'Д':
                item['answer_key'] = 'E'
            case _:
                print(f"Unknown answer key: {item['answer_key']}")
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Gold file saved to {path}")



def save_answers_to_json(answers: List[dict], filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(answers, f, indent=4)
    print(f"Answers saved to {filename}")

def extract_answer(raw: str) -> str:
    """
    Given a model output that ends with (or contains) the answer letter A–E,
    possibly wrapped in punctuation or markdown, return that letter.
    Returns an empty string if no valid answer is found.
    """
    # This looks for one of A–E, capturing it, followed only by non-letters/spaces to the end.
    m = re.search(r'([A-E])\W*$', raw.strip())
    return m.group(1) if m else ""

def extract_answer_bg(raw: str) -> str:
    """
    Given a model output that ends with (or contains) the answer letter А–Е,
    possibly wrapped in punctuation or markdown, return that letter.
    Returns an empty string if no valid answer is found.
    """
    # This looks for one of А–Е, capturing it, followed only by non-letters/spaces to the end.
    m = re.search(r'([А-Да-д]|[A-Ea-e])\W*$', raw.strip())
    return m.group(1) if m else ""

def postprocess_regex(answers):
    """
    Postprocesses the answers to extract the answer letter from the model output.
    """
    for ans in answers:
        ans["answer_key"] = extract_answer(ans["answer_key"][-5:])

def postprocess_regex_bg(answers):
    """
    Postprocesses the answers to extract the answer letter from the model output.
    """
    for ans in answers:
        ans["answer_key"] = extract_answer_bg(ans["answer_key"][-5:])