#!/usr/bin/env python3
"""
Шаг 1: обогащение датасета достопримечательностей
структурированными данными из открытых источников.

Назначение
-------------
Скрипт обрабатывает CSV-файл с достопримечательностями из Google Landmarks v2,
извлекает их Wikidata Q-ID и собирает расширенную информацию:
• Названия на нескольких языках (ru, en, de)
• Краткие описания из Wikidata
• Полные вводные абзацы из Wikipedia
• Координаты, год постройки, архитектурный стиль
• Статус объекта культурного наследия
• Ссылки на Wikipedia и официальные сайты
• Классификация по типу достопримечательности (музей, замок, церковь и т.д.)

Входные данные
-----------------
Файл: setup_data_v3/data/gl_human_made_approve_type.csv
Обязательные колонки:
  - landmark_id: уникальный идентификатор достопримечательности
  - category: URL категории Wikidata/Wikipedia (источник для поиска Q-ID)
  - hierarchical_label: иерархическая метка из Google Landmarks

Выходные данные
------------------
Файл: setup_data_v3/data/landmarks_data_wiki.json
Формат: JSON-массив объектов со следующей структурой:
{
  "landmark_id": "12345",
  "wikidata_id": "Q315493",
  "name_ru": "Маркграфский оперный театр",
  "name_en": "Margravial Opera House",
  "wikidata_description_ru": "барочный оперный театр в Байройте",
  "wikipedia_summary_ru": "Полный текст вводной секции из ru.wikipedia.org...",
  "coordinates": {"latitude": 49.9444, "longitude": 11.5753},
  "year_built": "1748",
  "country_ru": "Германия",
  "city_ru": "Байройт",
  "architectural_style_ru": ["барокко"],
  "heritage_status_ru": "Объект всемирного наследия ЮНЕСКО",
  "landmark_type": {"ru": "театр", "en": "theatre", "type_id": "театр"},
  "wikipedia_url_ru": "https://ru.wikipedia.org/wiki/...",
  "website": "https://example.com"
}

Требования:
------------------
    pip install aiohttp pandas tqdm
"""

import asyncio
import aiohttp
import re
import json
import time
import pandas as pd
from typing import Dict, List, Optional
from urllib.parse import unquote
from tqdm.asyncio import tqdm as atqdm
import logging

# ======================
# ЛОГИРОВАНИЕ
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class AsyncWikidataCollector:
    """Асинхронный сборщик с классификацией достопримечательностей"""
    
    SPARQL_URL = "https://query.wikidata.org/sparql"
    REST_URL = "https://www.wikidata.org/rest/v1"
    API_URL = "https://www.wikidata.org/w/api.php"
    WIKI_SUMMARY_URL = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
    
    # Типы достопримечательностей с Wikidata свойствами
    LANDMARK_TYPES = {
        "музей": {
            "ru": "музей",
            "en": "museum",
            "instance_of": ["Q33506", "Q207694"],  # музей, художественный музей
            "keywords": ["museum", "музей", "gallery", "галерея"]
        },
        "театр": {
            "ru": "театр",
            "en": "theatre",
            "instance_of": ["Q24354", "Q171603"],  # театр, театральное здание
            "keywords": ["theatre", "theater", "театр", "opera", "опера"]
        },
        "церковь": {
            "ru": "церковь",
            "en": "church",
            "instance_of": ["Q16970"],  # церковь
            "keywords": ["church", "церковь", "cathedral", "собор", "chapel", "часовня"]
        },
        "замок / крепость": {
            "ru": "замок / крепость",
            "en": "castle / fort",
            "instance_of": ["Q23413", "Q57831"],  # замок, крепость
            "keywords": ["castle", "замок", "fort", "крепость", "fortress", "цитадель"]
        },
        "руины": {
            "ru": "руины",
            "en": "ruins",
            "instance_of": ["Q109607"],  # руины
            "keywords": ["ruins", "руины", "archaeological", "археологический"]
        },
        "буддийский храм": {
            "ru": "буддийский храм",
            "en": "buddhist temple",
            "instance_of": ["Q5393308"],  # буддийский храм
            "keywords": ["buddhist", "буддийский", "temple", "храм", "pagoda", "пагода"]
        },
        "дом": {
            "ru": "дом",
            "en": "house",
            "instance_of": ["Q3947", "Q11755880"],  # дом, исторический дом
            "keywords": ["house", "дом", "mansion", "особняк", "villa", "вилла"]
        },
        "библиотека": {
            "ru": "библиотека",
            "en": "library",
            "instance_of": ["Q7075", "Q28564"],  # библиотека, публичная библиотека
            "keywords": ["library", "библиотека"]
        },
        "маяк": {
            "ru": "маяк",
            "en": "lighthouse",
            "instance_of": ["Q39715"],  # маяк
            "keywords": ["lighthouse", "маяк"]
        },
        "фонтан": {
            "ru": "фонтан",
            "en": "fountain",
            "instance_of": ["Q483453"],  # фонтан
            "keywords": ["fountain", "фонтан"]
        },
        "мост": {
            "ru": "мост",
            "en": "bridge",
            "instance_of": ["Q12280"],  # мост
            "keywords": ["bridge", "мост"]
        },
        "башня": {
            "ru": "башня",
            "en": "tower",
            "instance_of": ["Q12518"],  # башня
            "keywords": ["tower", "башня", "minaret", "минарет", "bell tower", "колокольня"]
        },
        "мечеть": {
            "ru": "мечеть",
            "en": "mosque",
            "instance_of": ["Q32815"],  # мечеть
            "keywords": ["mosque", "мечеть"]
        },
        "синтоистский храм": {
            "ru": "синтоистский храм",
            "en": "shinto shrine",
            "instance_of": ["Q946630"],  # синтоистский храм
            "keywords": ["shinto", "синтоистский", "shrine", "святилище"]
        },
        "дворец": {
            "ru": "дворец",
            "en": "palace",
            "instance_of": ["Q16560"],  # дворец
            "keywords": ["palace", "дворец"]
        },
        "мемориал": {
            "ru": "мемориал",
            "en": "memorial",
            "instance_of": ["Q5003624"],  # мемориал
            "keywords": ["memorial", "мемориал", "monument", "памятник"]
        },
        "монастырь": {
            "ru": "монастырь",
            "en": "monastery",
            "instance_of": ["Q44613"],  # монастырь
            "keywords": ["monastery", "монастырь", "abbey", "аббатство", "cloister", "клуатр"]
        },
        "небоскрёб": {
            "ru": "небоскрёб",
            "en": "skyscraper",
            "instance_of": ["Q41176"],  # небоскрёб
            "keywords": ["skyscraper", "небоскрёб", "high-rise", "высотка"]
        },
        "индуистский храм": {
            "ru": "индуистский храм",
            "en": "hindu temple",
            "instance_of": ["Q849570"],  # индуистский храм
            "keywords": ["hindu", "индуистский", "temple", "храм", "mandir"]
        },
        "синагога": {
            "ru": "синагога",
            "en": "synagogue",
            "instance_of": ["Q34627"],  # синагога
            "keywords": ["synagogue", "синагога"]
        },
        "обсерватория": {
            "ru": "обсерватория",
            "en": "observatory",
            "instance_of": ["Q62831"],  # обсерватория
            "keywords": ["observatory", "обсерватория"]
        },
        "правительственное здание": {
            "ru": "правительственное здание",
            "en": "government building",
            "instance_of": ["Q2659904"],  # правительственное здание
            "keywords": ["government", "правительственный", "parliament", "парламент", "capitol", "капитолий"]
        },
        "гостиница": {
            "ru": "гостиница",
            "en": "hotel",
            "instance_of": ["Q27686"],  # гостиница
            "keywords": ["hotel", "гостиница", "отель"]
        },
        "площадь": {
            "ru": "площадь",
            "en": "square",
            "instance_of": ["Q174782"],  # площадь
            "keywords": ["square", "площадь", "plaza"]
        },
        "археологический памятник": {
            "ru": "археологический памятник",
            "en": "archeological site",
            "instance_of": ["Q839954"],  # археологический памятник
            "keywords": ["archaeological", "археологический", "excavation", "раскопки"]
        },
        "ворота": {
            "ru": "ворота",
            "en": "gate",
            "instance_of": ["Q53060"],  # ворота
            "keywords": ["gate", "ворота", "portal", "портал"]
        },
        "скульптура": {
            "ru": "скульптура",
            "en": "sculpture",
            "instance_of": ["Q860861"],  # скульптура
            "keywords": ["sculpture", "скульптура", "statue", "статуя"]
        },
        "школа": {
            "ru": "школа",
            "en": "school",
            "instance_of": ["Q3914"],  # школа
            "keywords": ["school", "школа"]
        },
        "больница": {
            "ru": "больница",
            "en": "hospital",
            "instance_of": ["Q16917"],  # больница
            "keywords": ["hospital", "больница", "clinic", "клиника"]
        },
        "ветряная мельница": {
            "ru": "ветряная мельница",
            "en": "windmill",
            "instance_of": ["Q38720"],  # ветряная мельница
            "keywords": ["windmill", "ветряная мельница"]
        },
        "крест": {
            "ru": "крест",
            "en": "cross",
            "instance_of": ["Q40798"],  # крест
            "keywords": ["cross", "крест", "calvary", "голгофа"]
        },
        "пирамида": {
            "ru": "пирамида",
            "en": "pyramid",
            "instance_of": ["Q12516"],  # пирамида
            "keywords": ["pyramid", "пирамида"]
        }
    }
    
    # Известные соответствия для проблемных случаев
    KNOWN_ENTITIES = {
        "Markgräfliches Opernhaus": "Q315493",
        "Markgraefliches Opernhaus": "Q315493",
        "Markgräfliches Opernhaus Bayreuth": "Q315493",
        "Husa na provázku Theatre": "Q12020749",
        "Divadlo Husa na provázku": "Q12020749",
    }
    
    def __init__(self, user_agent: Optional[str] = None, delay: float = 1.0, debug: bool = False):
        self.user_agent = user_agent or "AIGuideBot/1.0 (slesarenko221999@gmail.com)"
        self.delay = delay
        self.debug = debug
        self.session: Optional[aiohttp.ClientSession] = None
        self.qid_cache: dict = {}
        self.label_cache: dict = {}
        # Счётчики для мониторинга rate-limit
        self.rate_limit_hits: int = 0
        self.total_requests: int = 0
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=20, ttl_dns_cache=300, force_close=False
        )
        self.session = aiohttp.ClientSession(
            headers={
                "User-Agent": self.user_agent,
                "From": "slesarenko221999@gmail.com",
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json"
            },
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=60)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _retry_request(self, url: str, method: str = "GET", 
                            params: Optional[dict] = None, data: Optional[bytes] = None, 
                            headers: Optional[dict] = None, max_retries: int = 3) -> Optional[aiohttp.ClientResponse]:
        """Retry-логика с экспоненциальной задержкой"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if method == "GET":
                    resp = await self.session.get(url, params=params, headers=headers)
                else:
                    resp = await self.session.post(url, params=params, data=data, headers=headers)
                
                if resp.status in [403, 429, 503]:
                    await resp.release()  # освобождаем соединение перед повтором
                    if resp.status == 429:
                        self.rate_limit_hits += 1
                    wait_time = (2 ** attempt) * self.delay
                    if self.debug:
                        print(
                            f"   ⏳ {resp.status} Error, retry in "
                            f"{wait_time:.1f}s (attempt {attempt+1})"
                            f" | 429 hits: {self.rate_limit_hits}"
                        )
                    await asyncio.sleep(wait_time)
                    continue
                self.total_requests += 1
                
                return resp
                
            except Exception as e:
                last_error = e
                if self.debug:
                    print(f"   ⚠️ Request error: {e}")
                await asyncio.sleep(self.delay * (attempt + 1))
        
        if self.debug and last_error:
            print(f"   ❌ All retries failed: {last_error}")
        return None
    
    async def extract_q_id(self, url: str) -> Optional[str]:
        """Извлекает Q-ID с улучшенным поиском по категориям"""
        url = url.strip()
        
        if url in self.qid_cache:
            if self.debug:
                print(f"   🔍 Using cached Q-ID: {self.qid_cache[url]}")
            return self.qid_cache[url]
        
        if 'wikidata.org/wiki/Q' in url:
            match = re.search(r'Q\d+', url)
            qid = match.group(0) if match else None
            if qid:
                self.qid_cache[url] = qid
            return qid
        
        if '/wiki/Category:' in url:
            category_name = url.split('/wiki/Category:')[-1].split('#')[0]
            category_name = unquote(category_name).replace('_', ' ').strip()
            
            if self.debug:
                print(f"   🔍 Searching for category: '{category_name}'")
            
            # Проверяем известные сущности
            for known_name, qid in self.KNOWN_ENTITIES.items():
                if known_name.lower() in category_name.lower() or category_name.lower() in known_name.lower():
                    if qid:
                        if self.debug:
                            print(f"   📚 Found in known entities: {known_name} -> {qid}")
                        self.qid_cache[url] = qid
                        return qid
            
            # Пробуем найти связанную статью в Wikipedia
            wiki_title = await self._find_wikipedia_article_from_category(category_name)
            if wiki_title:
                if self.debug:
                    print(f"   📚 Found Wikipedia article: {wiki_title}")
                qid = await self._get_qid_from_wikipedia_title(wiki_title)
                if qid:
                    self.qid_cache[url] = qid
                    return qid
            
            # Поиск напрямую в Wikidata (только en и ru)
            search_variants = self._generate_search_variants(category_name)
            for lang in ["en", "ru"]:
                for variant in search_variants:
                    qid = await self._search_entity_by_name(variant, lang)
                    if qid:
                        self.qid_cache[url] = qid
                        return qid
        
        return None
    
    def _generate_search_variants(self, name: str) -> List[str]:
        """Генерирует варианты названия для поиска"""
        variants = [name]
        
        # Убираем лишние слова
        cleaned = self._clean_category_name(name)
        if cleaned != name:
            variants.append(cleaned)
        
        # Обработка умлаутов
        if 'ü' in name:
            variants.append(name.replace('ü', 'ue'))
        if 'ä' in name:
            variants.append(name.replace('ä', 'ae'))
        if 'ö' in name:
            variants.append(name.replace('ö', 'oe'))
        if 'ß' in name:
            variants.append(name.replace('ß', 'ss'))
        
        return variants
    
    def _clean_category_name(self, name: str) -> str:
        """Очищает название категории"""
        stop_words = ['theatre', 'theater', 'cinema', 'building', 'category', 'the']
        words = name.split()
        cleaned = [w for w in words if w.lower() not in stop_words]
        return ' '.join(cleaned) if cleaned else name
    
    async def _find_wikipedia_article_from_category(self, category_name: str) -> Optional[str]:
        """Ищет статью в Wikipedia по категории (только en для снижения нагрузки)"""
        try:
            search_term = category_name.replace("Category:", "").strip()
            params = {
                "action": "query",
                "list": "search",
                "srsearch": search_term,
                "srlimit": 3,
                "format": "json",
            }
            url = "https://en.wikipedia.org/w/api.php"
            resp = await self._retry_request(url, params=params)
            if resp and resp.status == 200:
                data = await resp.json()
                for result in data.get("query", {}).get("search", []):
                    title = result.get("title", "")
                    if (title
                            and not title.startswith("Category:")
                            and not title.startswith("Wikipedia:")
                            and self._is_relevant_result(title, search_term)):
                        return title
        except Exception:
            pass
        return None
    
    def _is_relevant_result(self, title: str, search_term: str) -> bool:
        """Проверяет релевантность результата"""
        title_lower = title.lower()
        search_lower = search_term.lower()
        
        if title_lower == search_lower:
            return True
        
        if search_lower in title_lower or title_lower in search_lower:
            return True
        
        return False
    
    async def _get_qid_from_wikipedia_title(self, title: str, lang: str = "en") -> Optional[str]:
        """Получает Q-ID из Wikipedia статьи"""
        try:
            params = {
                "action": "query",
                "titles": title,
                "prop": "pageprops",
                "ppprop": "wikibase_item",
                "format": "json",
            }
            url = f"https://{lang}.wikipedia.org/w/api.php"
            resp = await self._retry_request(url, params=params)
            if resp and resp.status == 200:
                data = await resp.json()
                pages = data.get("query", {}).get("pages", {})
                for page_data in pages.values():
                    qid = page_data.get("pageprops", {}).get("wikibase_item")
                    if qid:
                        return qid
        except Exception:
            pass
        return None

    async def _search_entity_by_name(self, name: str, language: str = "en") -> Optional[str]:
        """Поиск сущности в Wikidata по названию (только en, limit=5)"""
        if not name:
            return None
        params = {
            "action": "wbsearchentities",
            "search": name,
            "language": language,
            "format": "json",
            "limit": 5,
            "type": "item",
        }
        try:
            resp = await self._retry_request(self.API_URL, params=params)
            if not resp or resp.status != 200:
                return None
            data = await resp.json()
            if not data.get("search"):
                return None
            # Точное совпадение
            for result in data["search"]:
                if result.get("label", "").lower() == name.lower():
                    return result.get("id")
            # Первый результат
            first = data["search"][0]
            if first.get("id", "").startswith("Q"):
                return first["id"]
        except Exception:
            pass
        return None
    
    async def fetch_entity_data(self, q_id: str) -> Optional[Dict]:
        """Получает данные о достопримечательности"""
        if self.debug:
            print(f"   🔍 Fetching data for {q_id}...")

        data = await self._fetch_wikidata_api(q_id)
        if data:
            data["landmark_type"] = await self._determine_landmark_type(data)
            data = await self._enrich_with_wiki_summaries(data)
            return data

        return None
    
    async def _determine_landmark_type(self, data: Dict) -> Optional[Dict]:
        """
        Определяет тип достопримечательности на основе данных из Wikidata
        Возвращает словарь с ru/en названиями типа
        """
        # Собираем все доступные данные для анализа
        description = (
            data.get("wikidata_description_ru", "") + " " +
            data.get("wikidata_description_en", "")
        ).lower()
        name = (
            data.get("name_ru", "") + " " +
            data.get("name_en", "")
        ).lower()
        
        # Проверяем по ключевым словам
        for type_key, type_info in self.LANDMARK_TYPES.items():
            # Проверяем ключевые слова в названии и описании
            for keyword in type_info["keywords"]:
                if keyword.lower() in name or keyword.lower() in description:
                    return {
                        "ru": type_info["ru"],
                        "en": type_info["en"],
                        "type_id": type_key
                    }
        
        # Если не нашли по ключевым словам, возвращаем общий тип
        return {
            "ru": "достопримечательность",
            "en": "landmark",
            "type_id": "landmark"
        }
    
    async def _fetch_wikidata_api(self, q_id: str) -> Optional[Dict]:
        """Получает данные через Wikidata Action API"""
        try:
            params = {
                "action": "wbgetentities",
                "ids": q_id,
                "props": "labels|descriptions|claims|sitelinks",
                "languages": "ru|en|de|fr|cs",
                "format": "json"
            }

            resp = await self._retry_request(self.API_URL, params=params)
            if not resp or resp.status != 200:
                return None

            data = await resp.json()
            entity = data.get("entities", {}).get(q_id, {})

            result = {
                "wikidata_id": q_id,
                "name_en": entity.get("labels", {}).get("en", {}).get("value"),
                "name_ru": entity.get("labels", {}).get("ru", {}).get("value"),
                "name_de": entity.get("labels", {}).get("de", {}).get("value"),
                # Краткие дизамбигуации из Wikidata (не полные описания)
                "wikidata_description_en": entity.get("descriptions", {}).get("en", {}).get("value"),
                "wikidata_description_ru": entity.get("descriptions", {}).get("ru", {}).get("value"),
            }

            claims = entity.get("claims", {})

            # Координаты (без сетевых запросов)
            if "P625" in claims:
                coords = self._extract_coordinates_from_claim(claims["P625"][0])
                if coords:
                    result["coordinates"] = coords

            # Год постройки / основания (без сетевых запросов)
            if "P571" in claims:
                year = self._extract_time_from_claim(claims["P571"][0])
                if year:
                    result["year_built"] = year

            # Собираем все entity ID для батчевого запроса меток
            country_id = (
                self._extract_entity_id_from_claim(claims["P17"][0])
                if "P17" in claims else None
            )
            city_id = (
                self._extract_entity_id_from_claim(claims["P131"][0])
                if "P131" in claims else None
            )
            style_ids = [
                self._extract_entity_id_from_claim(c)
                for c in claims.get("P149", [])[:3]
            ]
            style_ids = [s for s in style_ids if s]
            arch_ids = [
                self._extract_entity_id_from_claim(c)
                for c in claims.get("P84", [])[:3]
            ]
            arch_ids = [a for a in arch_ids if a]
            heritage_id = (
                self._extract_entity_id_from_claim(claims["P1435"][0])
                if "P1435" in claims else None
            )

            all_ids = list(filter(None, [
                country_id, city_id, heritage_id, *style_ids, *arch_ids
            ]))

            # Два батчевых запроса вместо N последовательных
            labels_ru = await self._get_entity_labels_batch(all_ids, lang="ru")
            labels_en = await self._get_entity_labels_batch(all_ids, lang="en")

            if country_id:
                result["country_ru"] = labels_ru.get(country_id)
                result["country_en"] = labels_en.get(country_id)
            if city_id:
                result["city_ru"] = labels_ru.get(city_id)
                result["city_en"] = labels_en.get(city_id)
            if heritage_id:
                result["heritage_status_ru"] = labels_ru.get(heritage_id)
                result["heritage_status_en"] = labels_en.get(heritage_id)
            if style_ids:
                result["architectural_style_ru"] = [
                    labels_ru[s] for s in style_ids if labels_ru.get(s)
                ]
                result["architectural_style_en"] = [
                    labels_en[s] for s in style_ids if labels_en.get(s)
                ]
            if arch_ids:
                result["architect_ru"] = [
                    labels_ru[a] for a in arch_ids if labels_ru.get(a)
                ]
                result["architect_en"] = [
                    labels_en[a] for a in arch_ids if labels_en.get(a)
                ]

            # Изображение (P18) — имя файла на Wikimedia Commons
            # if "P18" in claims:
            #     try:
            #         img_snak = claims["P18"][0].get("mainsnak", {})
            #         if img_snak.get("snaktype") == "value":
            #             img_val = img_snak.get("datavalue", {}).get("value", "")
            #             if img_val:
            #                 img_name = img_val.replace(" ", "_")
            #                 result["image_url"] = (
            #                     f"https://commons.wikimedia.org/wiki/Special:FilePath/{img_name}"
            #                 )
            #     except Exception:
            #         pass

            # Официальный сайт (P856)
            if "P856" in claims:
                try:
                    web_snak = claims["P856"][0].get("mainsnak", {})
                    if web_snak.get("snaktype") == "value":
                        website = web_snak.get("datavalue", {}).get("value", "")
                        if website:
                            result["website"] = website
                except Exception:
                    pass

            # Wikipedia ссылки
            sitelinks = entity.get("sitelinks", {})
            if "enwiki" in sitelinks:
                title = sitelinks["enwiki"]["title"].replace(" ", "_")
                result["wikipedia_url_en"] = f"https://en.wikipedia.org/wiki/{title}"
            if "ruwiki" in sitelinks:
                title = sitelinks["ruwiki"]["title"].replace(" ", "_")
                result["wikipedia_url_ru"] = f"https://ru.wikipedia.org/wiki/{title}"
            if "dewiki" in sitelinks:
                title = sitelinks["dewiki"]["title"].replace(" ", "_")
                result["wikipedia_url_de"] = f"https://de.wikipedia.org/wiki/{title}"

            return {k: v for k, v in result.items() if v not in [None, "", [], {}]}

        except Exception as e:
            if self.debug:
                print(f"   ⚠️ API error: {e}")
        return None
    
    def _extract_entity_id_from_claim(self, claim: Dict) -> Optional[str]:
        """Извлекает ID сущности из claim"""
        try:
            mainsnak = claim.get("mainsnak", {})
            if mainsnak.get("snaktype") != "value":
                return None
            datavalue = mainsnak.get("datavalue", {})
            if datavalue.get("type") != "wikibase-entityid":
                return None
            value = datavalue.get("value", {})
            return value.get("id") if isinstance(value, dict) else None
        except Exception:
            return None

    def _extract_time_from_claim(self, claim: Dict) -> Optional[str]:
        """Извлекает год из claim. Поддерживает отрицательные годы (до н.э.)."""
        try:
            mainsnak = claim.get("mainsnak", {})
            if mainsnak.get("snaktype") != "value":
                return None
            datavalue = mainsnak.get("datavalue", {})
            if datavalue.get("type") != "time":
                return None
            value = datavalue.get("value", {})
            time_val = value.get("time")
            if time_val:
                # Формат Wikidata: +1889-00-00T... или -0447-00-00T...
                match = re.match(r'^([+-]?)(\d+)', time_val)
                if match:
                    sign = match.group(1)
                    year = int(match.group(2))
                    if sign == "-" and year > 0:
                        return f"{year} до н.э."
                    return str(year)
        except Exception:
            pass
        return None

    def _extract_coordinates_from_claim(self, claim: Dict) -> Optional[Dict]:
        """Извлекает координаты из claim"""
        try:
            mainsnak = claim.get("mainsnak", {})
            if mainsnak.get("snaktype") != "value":
                return None
            datavalue = mainsnak.get("datavalue", {})
            if datavalue.get("type") != "globecoordinate":
                return None
            value = datavalue.get("value", {})
            if value.get("latitude") and value.get("longitude"):
                return {
                    "latitude": float(value["latitude"]),
                    "longitude": float(value["longitude"])
                }
        except Exception:
            pass
        return None
    
    async def _get_entity_labels_batch(
        self, entity_ids: List[str], lang: str = "ru"
    ) -> Dict[str, Optional[str]]:
        """Батчевый запрос меток для нескольких entity ID за один запрос.

        Wikidata API поддерживает до 50 ID через '|'.
        Возвращает словарь {entity_id: label}.
        """
        if not entity_ids:
            return {}

        # Фильтруем уже закэшированные
        def cache_key_fn(eid: str) -> str:
            return f"{eid}_{lang}"

        missing = [
            eid for eid in entity_ids
            if cache_key_fn(eid) not in self.label_cache
        ]
        result: Dict[str, Optional[str]] = {
            eid: self.label_cache[cache_key_fn(eid)]
            for eid in entity_ids
            if cache_key_fn(eid) in self.label_cache
        }

        if not missing:
            return result

        # Запрашиваем батчами по 50 (лимит Wikidata API)
        for i in range(0, len(missing), 50):
            batch = missing[i: i + 50]
            try:
                params = {
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "labels",
                    "languages": f"{lang}|en",
                    "format": "json",
                }
                resp = await self._retry_request(self.API_URL, params=params)
                if resp and resp.status == 200:
                    data = await resp.json()
                    entities = data.get("entities", {})
                    for eid in batch:
                        labels = entities.get(eid, {}).get("labels", {})
                        label = (
                            labels.get(lang, {}).get("value")
                            or labels.get("en", {}).get("value")
                        )
                        self.label_cache[cache_key_fn(eid)] = label
                        result[eid] = label
            except Exception:
                for eid in batch:
                    result[eid] = None

        return result

    async def _get_entity_label(self, entity_id: str, lang: str = "ru") -> Optional[str]:
        """Получает метку одной сущности (использует батчевый кэш)."""
        if not entity_id:
            return None
        labels = await self._get_entity_labels_batch([entity_id], lang=lang)
        return labels.get(entity_id)
    
    async def _get_wikipedia_summary(
        self,
        url: Optional[str],
        lang: str,
        sentences: int = 10
    ) -> Optional[str]:
        """Получает расширенное описание из Wikipedia через Action API
        
        Использует TextExtracts API для получения нескольких абзацев.
        
        Args:
            url: URL статьи Wikipedia
            lang: язык (ru, en, de и т.д.)
            sentences: количество предложений (10 = ~2-3 абзаца)
        
        Returns:
            Текст описания или None
        
        Параметры API:
        - exintro: только вводная секция (до первого заголовка)
        - explaintext: plain text без HTML
        - exsentences: количество предложений
        """
        if not url:
            return None
        
        try:
            title = url.split("/")[-1].strip()
            
            # Используем Action API с TextExtracts
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "exintro": "1",  # только вводная секция
                "explaintext": "1",  # plain text
                "exsentences": str(sentences),
            }
            
            api_url = f"https://{lang}.wikipedia.org/w/api.php"
            resp = await self._retry_request(api_url, params=params)
            
            if resp and resp.status == 200:
                data = await resp.json()
                pages = data.get("query", {}).get("pages", {})
                
                # Берем первую (и единственную) страницу
                for page_data in pages.values():
                    extract = page_data.get("extract", "").strip()
                    if extract and len(extract) > 50:  # минимальная длина
                        return extract
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ Wikipedia summary error for {lang}: {e}")
        return None
    
    async def _get_wikipedia_full_intro(
        self,
        url: Optional[str],
        lang: str
    ) -> Optional[str]:
        """Получает полную вводную секцию из Wikipedia (без ограничения)
        
        Возвращает весь текст до первого заголовка раздела.
        Полезно для получения максимально полного описания.
        
        Args:
            url: URL статьи Wikipedia
            lang: язык (ru, en, de и т.д.)
        
        Returns:
            Полный текст вводной секции или None
        """
        if not url:
            return None
        
        try:
            title = url.split("/")[-1].strip()
            
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "exintro": "1",  # только вводная секция
                "explaintext": "1",  # plain text
                # Без exsentences - получаем всю вводную секцию
            }
            
            api_url = f"https://{lang}.wikipedia.org/w/api.php"
            resp = await self._retry_request(api_url, params=params)
            
            if resp and resp.status == 200:
                data = await resp.json()
                pages = data.get("query", {}).get("pages", {})
                
                for page_data in pages.values():
                    extract = page_data.get("extract", "").strip()
                    if extract and len(extract) > 100:
                        return extract
        except Exception as e:
            if self.debug:
                print(f"   ⚠️ Wikipedia full intro error for {lang}: {e}")
        return None
    
    async def _enrich_with_wiki_summaries(
        self,
        data: Dict,
        use_full_intro: bool = False,
        sentences: int = 10
    ) -> Dict:
        """Добавляет Wikipedia summaries
        
        Args:
            data: словарь с данными достопримечательности
            use_full_intro: если True, получает полную вводную секцию
                           если False, ограничивает количеством предложений
            sentences: количество предложений (если use_full_intro=False)
        
        Returns:
            Обогащенный словарь с полями wikipedia_summary_{lang}
        """
        wiki_fields = [
            ("wikipedia_url_en", "en"),
            ("wikipedia_url_ru", "ru"),
            ("wikipedia_url_de", "de")
        ]
        
        for url_key, lang in wiki_fields:
            if url_key in data:
                if use_full_intro:
                    summary = await self._get_wikipedia_full_intro(
                        data[url_key], lang
                    )
                else:
                    summary = await self._get_wikipedia_summary(
                        data[url_key], lang, sentences=sentences
                    )
                
                if summary:
                    data[f"wikipedia_summary_{lang}"] = summary
        
        return data
    
    async def collect_batch(
        self,
        rows: List[Dict],
        concurrency: int = 2,
        checkpoint_file: Optional[str] = None,
        checkpoint_every: int = 50,
        existing_results: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Сбор данных с контролем параллелизма и checkpoint-сохранением.

        Args:
            rows: список словарей с ключами 'category' (URL),
                  'landmark_id' и 'hierarchical_label'.
            concurrency: максимальное число параллельных запросов.
            checkpoint_file: путь к файлу для промежуточного сохранения.
                Если None — checkpoint отключён.
            checkpoint_every: сохранять checkpoint каждые N записей в батче.
            existing_results: существующие результаты из чекпоинта.
        """
        results: List[Dict] = existing_results.copy() if existing_results else []
        semaphore = asyncio.Semaphore(concurrency)
        success_count = len([r for r in results if "error" not in r])
        pbar = atqdm(total=len(rows), desc="Collecting", unit="landmark")

        async def process_one(row: Dict) -> Dict:
            nonlocal success_count
            url = row["category"]
            landmark_id = row.get("landmark_id")
            hierarchical_label = row.get("hierarchical_label")

            async with semaphore:
                if self.debug:
                    pbar.write(f"📍 [{landmark_id}] {url}")

                q_id = await self.extract_q_id(url)
                if not q_id:
                    pbar.write(f"⚠️  Q-ID not found: {url}")
                    pbar.update(1)
                    return {
                        "landmark_id": landmark_id,
                        "hierarchical_label": hierarchical_label,
                        "url": url,
                        "error": "Q-ID not found",
                    }

                if self.debug:
                    pbar.write(f"   ✓ Q-ID: {q_id}")

                data = await self.fetch_entity_data(q_id)
                rl = self.rate_limit_hits
                if data:
                    success_count += 1
                    data["url"] = url
                    data["landmark_id"] = landmark_id
                    data["hierarchical_label"] = hierarchical_label
                    name = data.get("name_ru") or data.get("name_en", "N/A")
                    ltype = data.get("landmark_type", {}).get("ru", "?")
                    postfix = f"✅ {success_count} | {name} [{ltype}]"
                    if rl:
                        postfix += f" | 429×{rl}"
                    pbar.set_postfix_str(postfix, refresh=True)
                    pbar.update(1)
                    return data
                else:
                    msg = f"⚠️  No Wikidata entity for {q_id}: {url}"
                    if rl:
                        msg += f" | 429×{rl}"
                    pbar.write(msg)
                    pbar.update(1)
                    return {
                        "landmark_id": landmark_id,
                        "hierarchical_label": hierarchical_label,
                        "url": url,
                        "wikidata_id": q_id,
                        "error": "No Wikidata entity",
                    }

        # Обрабатываем батчами для поддержки checkpoint
        batch_size = concurrency * checkpoint_every
        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start: batch_start + batch_size]
            tasks = [process_one(row) for row in batch]
            raw = await asyncio.gather(*tasks, return_exceptions=True)

            for r in raw:
                if isinstance(r, Exception):
                    pbar.write(f"⚠️  Exception: {r}")
                else:
                    results.append(r)

            # Checkpoint после каждого батча
            if checkpoint_file and results:
                success = [r for r in results if "error" not in r]
                self.save_json(success, checkpoint_file)
                pbar.write(
                    f"💾 Checkpoint: {len(success)} → {checkpoint_file}"
                )

        pbar.close()
        return results

    def save_json(self, data: List[Dict], filename: str):
        """Сохраняет результаты в JSON"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved: {filename}")


# ==================== ТЕСТОВЫЙ ЗАПУСК ====================

async def main():
    csv_path = (
        "setup_data_v3/data/gl_human_made_approve_type.csv"
    )
    checkpoint_file = "setup_data_v3/data/landmarks_data_wiki_checkpoint.json"
    
    df = pd.read_csv(csv_path)
    # Преобразуем DataFrame в список словарей
    rows = df[["landmark_id", "category", "hierarchical_label"]].to_dict(
        orient="records"
    )
    
    # Загружаем существующий чекпоинт, если есть
    existing_results = []
    processed_ids = set()
    
    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
            processed_ids = {r.get("landmark_id") for r in existing_results if r.get("landmark_id")}
            print(f"📂 Загружен чекпоинт: {len(existing_results)} записей")
            print(f"   Обработано landmark_id: {len(processed_ids)}")
    except FileNotFoundError:
        print("📂 Чекпоинт не найден, начинаем с начала")
    
    # Фильтруем только необработанные записи
    rows_to_process = [r for r in rows if r.get("landmark_id") not in processed_ids]
    
    print(f"📊 Всего записей: {len(rows)}")
    print(f"✅ Уже обработано: {len(processed_ids)}")
    print(f"⏳ Осталось обработать: {len(rows_to_process)}")
    
    if not rows_to_process:
        print("✨ Все записи уже обработаны!")
        return existing_results

    async with AsyncWikidataCollector(
        user_agent="LandmarkCollector/1.0 (slesarenko221999@gmail.com)",
        delay=1.0,   # задержка при retry 429/503
        debug=False,
    ) as collector:

        print(f"\n🚀 Продолжаем сбор данных для {len(rows_to_process)} landmarks...")
        start = time.time()

        all_results = await collector.collect_batch(
            rows_to_process,
            concurrency=2,  # 2 параллельных — безопасно для Wikidata
            checkpoint_file=checkpoint_file,
            checkpoint_every=100,
            existing_results=existing_results,
        )

        elapsed = time.time() - start
        success = [r for r in all_results if "error" not in r]
        failed = len(all_results) - len(success)

        logger.info("=" * 60)
        logger.info("ШАГ 1: СБОР ОПИСАНИЙ ЗАВЕРШЕН")
        logger.info("=" * 60)
        print(
            f"\n📊 Итого: {len(success)} успешно, {failed} с ошибками"
            f" | Время обработки новых: {elapsed:.1f}s"
        )

        # Группируем по типам
        types_count: Dict[str, int] = {}
        for item in success:
            ltype = item.get("landmark_type", {}).get("ru", "неизвестно")
            types_count[ltype] = types_count.get(ltype, 0) + 1

        print("\n📊 Распределение по типам:")
        for type_name, count in sorted(types_count.items(), key=lambda x: x[1], reverse=True):
            print(f"   {type_name}: {count}")

        if success:
            collector.save_json(success, "setup_data_v3/data/landmarks_data_wiki.json")

        return all_results


if __name__ == "__main__":
    asyncio.run(main())