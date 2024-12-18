from fastapi import APIRouter
from schemas.wikipedia_schemas import MunichWiki
from models.wikipedia import WikipediaMunich

router = APIRouter()
wm= WikipediaMunich()

@router.post("/get_munich")
async def get_munich(request:MunichWiki):
    try:
        document = wm.fetch_wikipages(page=request.page_head)
    except Exception as e:
        answer= {
            'status_code': 400,
            'status': 'error',
            'result': [str(e)],
            'message':'ERROR[fetch_wikipages]'
        }
        return answer
    try:
        chunks=wm.document_splitter(document=document)
    except Exception as e:
        answer= {
            'status_code': 400,
            'status': 'error',
            'result': [str(e)],
            'message':'ERROR[document_splitter]'
        }
        return answer
    try:
        answer1 = wm.query_munich_page(query=request.question,
                                        chunks=chunks)
        answer= {
            'status_code': 200,
            'status': 'success',
            'result': answer1,
            'message':'Successfully extracted answer for the question'
        }
        return answer
    except Exception as e:
        answer= {
            'status_code': 400,
            'status': 'error',
            'result': [str(e)],
            'message':'ERROR[document_splitter]'
        }
        return answer
