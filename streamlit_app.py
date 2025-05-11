from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json
import re
from dotenv import load_dotenv
import logging
import streamlit as st


load_dotenv()
logging.basicConfig(level=logging.INFO)

class UnusualRequirement(BaseModel):
    requirement: str = Field(description="Нестандартное или оригинальное требование клиента")
    category: Optional[str] = Field(
        description="Категория требования (локация, питание, вид, активности и т.д.)", 
        default=None
    )

class ExtractionResult(BaseModel):
    unusual_requirements: List[UnusualRequirement] = Field(
        description="Список нестандартных требований из запроса клиента"
    )

template = """
Извлеки и сохрани нестандартные требования, упомянутые в запросе к туроператору.

Нестандартные требования - это особые пожелания, которые выходят за рамки обычного запроса на тур.
Примеры нестандартных требований:
- Отель рядом с местом проведения Формулы 1 (категория: локация)
- Отель с собственным пляжем (категория: локация)
- Особенное отношение к гостям (категория: сервис)
- Особенный уровень проживания (категория: сервис)
- С инфраструктурой для детей (категория: сервис)
- Можно заселиться с животными (категория: сервис)
- Размещение с видом на определенную достопримечательность (категория: вид)
- Размещение со специфическими требованиями к виду (категория: вид)
- Близость к месту проведения фестиваля или концерта (категория: локация)
- Специфические требования к питанию (категория: питание)
- Необычные экскурсии или активности (категория: активности)
- Особые требования к транспорту (категория: транспорт)
- Прогулки на катере/вертолете/дельтаплане и так далее (категория: транспорт)
- Сложные маршруты с несколькими городами (категория: маршрут)

НЕ СЧИТАЮТСЯ нестандартными требованиями, например:
- Количество человек (взрослых/детей)
- Стандартные даты заезда/выезда
- Обычные типы размещения (SGL, DBL, TWIN, и т.д.)
- Обычные типы питания (завтрак, полупансион)
- Стандартные трансферы из аэропорта в отель

Текст запроса: {input}

Если нет нестандартных требований, верни пустой список.
"""

prompt = ChatPromptTemplate.from_template(template)

def clean_html(text):
    return re.sub(r'<.*?>', ' ', text).replace('  ', ' ').strip()

llm = ChatOpenAI(model="gpt-4.1", temperature=0)
extraction_chain = prompt | llm.with_structured_output(ExtractionResult)

def extract_unusual_requirements(requests_json):
    results = []
    
    for request in requests_json[:500]:
        try:
            comment = request.get("Comment", "")
            if comment:
                clean_comment = clean_html(comment)
                
                extraction_result = extraction_chain.invoke({"input": clean_comment})
                unusual_reqs = [
                    {"requirement": req.requirement, "category": req.category}
                    for req in extraction_result.unusual_requirements
                ]
                
                result = {
                    "number": request.get("number"),
                    "original_comment": comment,
                    "unusual_requirements": unusual_reqs
                }
                yield result
            else:
                yield {
                    "number": request.get("number"),
                    "unusual_requirements": []
                }
        except Exception as e:
            yield {
                "number": request.get("number"),
                "error": str(e)
            }
    
    return results



st.set_page_config(
    page_title="Нестандартные требования",
    page_icon="🦄",
    layout="wide"
)

st.markdown(
    """
    <style>
    .result-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .req-list li {
        margin-bottom: 0.3em;
    }
    .request-block {
        background-color: #f7f7fc;
        border-radius: 8px;
        padding: 1em 1.5em;
        margin-bottom: 1.5em;
        box-shadow: 0 2px 8px rgba(100,125,222,0.09);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Извлечение нестандартных требований")
st.caption("Загрузите ваш JSON-файл с заявками, чтобы увидеть найденные необычные требования по каждой из них.")

uploaded_file = st.file_uploader("Загрузите JSON-файл", type=["json"])
if uploaded_file:
    requests_data = json.load(uploaded_file)
    st.info("Обработка...")

    progress_bar = st.progress(0)
    total = min(10, len(requests_data))
    results = []

    for idx, result in enumerate(extract_unusual_requirements(requests_data), 1):
        with st.container():
            #st.markdown(f'<div class="request-block">', unsafe_allow_html=True)
            st.markdown(f'<span class="result-header">🔹 {idx}. Заявка № {result["number"]}</span>', unsafe_allow_html=True)
            with st.expander("Показать исходный запрос"):
                st.markdown(f'<div style="color:#444;">{result["original_comment"] or "*Комментарий отсутствует*"}</div>', unsafe_allow_html=True)
            if "error" in result:
                st.error(f"Ошибка при обработке: {result['error']}")
            elif result["unusual_requirements"]:
                st.markdown("**Найденные необычные требования:**")
                st.markdown('<ul class="req-list">', unsafe_allow_html=True)
                for req in result["unusual_requirements"]:
                    st.markdown(f'<li> <b>{req["category"]}</b>: {req["requirement"]} </li>', unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)
            else:
                st.info("Необычных требований не найдено.")
            st.markdown('</div>', unsafe_allow_html=True)
        results.append(result)
        progress_bar.progress(idx / total)

    st.success("✅ Обработка завершена!")

    st.download_button(
        "💾 Скачать результаты",
        data=json.dumps(results, ensure_ascii=False, indent=2),
        file_name="unusual_requirements.json",
        mime="application/json"
    )
