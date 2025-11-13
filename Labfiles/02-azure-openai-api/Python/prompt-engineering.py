import os
import asyncio
from dotenv import load_dotenv

# Azure OpenAI package 추가
from openai import AsyncAzureOpenAI

# 매번 전체 응답을 출력할지 여부
printFullResponse = False

# 동시에 처리할 task 수 설정
num_tasks = 1

async def main():
    try:
        # Azure OpenAI 설정
        load_dotenv()
        azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
        azure_oai_key = os.getenv("AZURE_OAI_KEY")
        azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
        
        # Azure OpenAI client 설정
        client = AsyncAzureOpenAI(
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            api_version="2025-01-01-preview"
        )

        while True:
            # 사용자의 system prompt 수정 또는 확인을 위한 대기
            print("---------------\nPausing the app to allow you to change the system prompt.\nPress enter to continue...")
            input()

            # System message 읽기 및 사용자 prompt 입력 받기
            system_text = open(file="system.txt", encoding="utf8").read().strip()
            user_text = input("Enter user message, or 'quit' to exit: ")
            if user_text.lower() == 'quit' or system_text.lower() == 'quit':
                print('Exiting program...')
                break
            
            # Ground content 읽기 및 사용자 prompt에 ground content 추가
            print("\nAdding grounding context from grounding.txt")
            ground_text = open(file="grounding.txt", encoding="utf8").read().strip()
            full_user_text = user_text + ground_text
            
            # 동시에 여러 요청(task) 생성
            tasks = []
            for i in range(num_tasks):
                # 요청마다 구분하기 위해 문구 추가 (제거 가능)
                task_text = full_user_text + f"\n(Request {i+1})"
                task = asyncio.create_task(
                    call_openai_model(system_message=system_text,
                                      user_message=task_text,
                                      model=azure_oai_deployment,
                                      client=client)
                )
                tasks.append(task)
            
            # 응답이 도착하는 순서대로 처리
            for completed in asyncio.as_completed(tasks):
                result = await completed  # result에는 생성된 텍스트가 담겨 있음
                print("Processed answer:\n" + result + "\n")
    
    except Exception as ex:
        print(ex)

async def call_openai_model(system_message, user_message, model, client):
    # Message 포매팅
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    
    print("\nSending request to Azure OpenAI model for user message:\n" + user_message + "\n")
    
    # 모델에 요청 송부
    response = await client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=800,
        messages=messages
    )
    
    # 생성된 텍스트 추출
    generated_text = response.choices[0].message.content
    #messages.append({"role": "assistant", "content": generated_text})
    
    if printFullResponse:
        print("Full response object:")
        print(response)
    
    return generated_text

if __name__ == '__main__':
    asyncio.run(main())