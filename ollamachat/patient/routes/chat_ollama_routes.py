from fastapi import APIRouter
from schemas.chat_ollama_schemas import LaamaChat
from models.chat_ollama import OllaChat

router = APIRouter()
oc = OllaChat()

@router.post('/get_patient_details')
async def get_patient_details(request:LaamaChat):
    try:
        response = oc.chat_llama(user_input=request.user_input)
        print(response)
        answer= {
            'status_code': 200,
            'status': 'success',
            'result': response,
            'message':'Successfully extracted user details from the provided Text'
        }
        return answer
    except Exception as e:
        answer= {
            'status_code': 400,
            'status': 'error',
            'result':response,
            'message':'ERROR[chat_llama]'
        }
        return answer