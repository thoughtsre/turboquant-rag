import asyncio
import logging
import time
from nicegui import ui
from lib.turboquant.turboquant import TurboQuantRAG

logger = logging.getLogger(__name__)

CODEBOOK_PATH = "./data/codebook/384d_codebook.pkl"
DOCS_PATH = "./data/raw/arxiv-metadata-oai-snapshot.parquet"
EMBEDDINGS_PATH = "./data/processed/arxiv_quantized_embeddings_4bits_seed=42_qjl_seed=24.parquet"


rag = TurboQuantRAG(
    4,
    CODEBOOK_PATH,
    DOCS_PATH,
    EMBEDDINGS_PATH,
)


def root():
    ui.add_css("./static/style.css")

    dialog = ui.dialog()

    def open_modal(row):
        dialog.clear()
        with dialog, ui.card().classes('max-w-2xl'):
            ui.label(row['title'].strip()).classes('text-xl font-bold')
            ui.label(f"Authors: {row['authors']}").classes('text-sm text-grey-7 italic')
            ui.label(row['abstract'].strip()).classes('text-sm whitespace-pre-line')
            ui.link('View on arXiv →', f"https://arxiv.org/abs/{row['id']}", new_tab=True) \
                .classes('text-teal-700 mt-2')
            ui.button('Close', on_click=dialog.close).props('flat')
        dialog.open()

    def render_card(row):
        abstract = row['abstract'] or ''
        truncated = abstract.strip().replace('\n', ' ')
        if len(truncated) > 280:
            truncated = truncated[:280].rsplit(' ', 1)[0] + '…'
        with ui.card().classes('w-full cursor-pointer hover:shadow-lg') \
                .on('click', lambda r=row: open_modal(r)):
            ui.label(row['title'].strip()).classes('text-lg font-bold')
            ui.label(truncated).classes('text-sm text-grey-8')

    async def on_search():
        q = search_input.value
        if not q:
            return
        results_container.clear()
        spinner.classes(remove='hidden')
        t0 = time.perf_counter()
        df = await asyncio.to_thread(rag.query, q, 12)
        elapsed = time.perf_counter() - t0
        spinner.classes(add='hidden')
        with results_container:
            ui.label(f"Searched {rag.n_docs} documents in {elapsed:.2f}s").classes('text-xs text-grey-6')
            for row in df.iter_rows(named=True):
                render_card(row)

    with ui.row(align_items="stretch").classes('w-full'):
        ui.markdown("## thoughtsre: TurboQuant Document Retrieval").classes('text-teal-700')

    with ui.row(align_items="stretch").classes('w-full'):
        search_input = ui.input(placeholder="Enter your query here...").props('rounded outlined input-class=mx-3 clearable').classes('flex-grow')
        search_button = ui.button("Search", color="#8fbcb5").props('rounded')

    spinner = ui.spinner().classes('hidden')
    results_container = ui.column().classes('w-full gap-4')

    search_button.on_click(on_search)
    search_input.on('keydown.enter', on_search)


if __name__ in {"__main__", "__mp_main__"}:

    ui.run(root)
