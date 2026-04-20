# Pendientes — LEC Retrieval Platform

## Código por escribir

| Archivo | Descripción |
|---|---|
| `eval/generate_qas.py` | Lee chunks de MongoDB, usa LLM para generar 20+ pares query/answer, guarda en `eval/qas.json` |
| `eval/qas.json` | Se genera con el script anterior — necesita corpus indexado primero |
| `eval/evaluate.py` | Corre los 3 modos (semantic / hybrid / hybrid+rerank), mide precision@5, recall@5, NDCG |

## Proceso pendiente (en orden)

1. Esperar que termine la descarga actual (`tail -f /tmp/lec_ingest.log`)
2. Re-correr `python -m src.ingesta.ingest` — descarga las 30 queries nuevas (laborales + otras) para llegar a 1,000+ docs
3. Correr `python create_vector_index.py` — crea índices vectorial y BM25 en MongoDB Atlas (solo una vez)
4. Verificar que `index_corpus()` indexó todo en MongoDB
5. Correr `python eval/generate_qas.py` — genera los QA pairs
6. Correr `python eval/evaluate.py` — obtiene las métricas del assignment

## Documentos del assignment (obligatorios para submit)

| Archivo | Descripción |
|---|---|
| `README.md` | Setup, variables de entorno, cómo correr cada paso |
| `report.md` | ≤2 páginas: qué construiste, qué rompió, qué aprendiste |
| `architecture.md` | Stack elegido, alternativas rechazadas, trade-offs |
| `roadmap.md` | 3-5 features para la próxima semana con justificación |
| `ai_usage.md` | Qué escribió Claude, qué es tuyo, cómo verificaste |

## Bugs conocidos

- `manifest.csv` acumula entradas duplicadas entre runs — no es crítico pero ensució el conteo
- Necesita `pytest.ini` para configurar `pytest-asyncio` en modo `auto`

## Comandos clave

```bash
# Ver progreso de descarga
tail -f /tmp/lec_ingest.log

# Re-correr ingesta (incremental)
python -m src.ingesta.ingest

# Crear índices Atlas (una vez)
python create_vector_index.py

# Correr app
streamlit run streamlit/app.py
uvicorn api.main:app --reload

# Tests
pytest tests/test.py -v
```

## Estado actual del código

| Archivo | Estado |
|---|---|
| `src/utils/mongodb.py` | ✅ Listo |
| `src/utils/embeddings.py` | ✅ Listo |
| `src/utils/llm.py` | ✅ Listo |
| `src/ingesta/ingest.py` | ✅ Listo (50 queries) |
| `src/retrieval/retrieval.py` | ✅ Listo |
| `src/generate/generate.py` | ✅ Listo |
| `api/main.py` + `api/models.py` | ✅ Listo |
| `streamlit/app.py` | ✅ Listo |
| `tests/test.py` | ✅ Listo |
| `create_vector_index.py` | ✅ Listo |
| `Dockerfile` | ✅ Listo |
| `eval/generate_qas.py` | ❌ Pendiente |
| `eval/evaluate.py` | ❌ Pendiente |
| `README.md` | ❌ Pendiente |
| `report.md` | ❌ Pendiente |
| `architecture.md` | ❌ Pendiente |
| `roadmap.md` | ❌ Pendiente |
| `ai_usage.md` | ❌ Pendiente |