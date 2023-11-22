from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

import asyncio
from dotenv import load_dotenv;load_dotenv() # openai_key  .env 선언 사용 


from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tiktoken_len(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)


loader = TextLoader("files\jnewspaper.txt", encoding='utf-8')
documents = loader.load()
# print("Page Content:\n", documents[0].page_content)
# print("\nMetadata:", documents[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =50,
        chunk_overlap  = 0,
        separators=["\n"],
        length_function =tiktoken_len
    )

pages = text_splitter.split_documents(documents)
print( len(pages) )
i=0
for p in pages:
    i=i+1
    print( "{:02d} {}".format(i, tiktoken_len(p.page_content)), p.page_content.replace('\n', ''), p.metadata['source'])

print("="*100)
index = FAISS.from_documents(pages , OpenAIEmbeddings())

index.save_local("faiss-newspaper")

# query = "현재 교장은?"
# # docs = index.similarity_search(query) 유사도가 없다.
# loop = asyncio.get_event_loop()
# docs = loop.run_until_complete( index.asimilarity_search_with_relevance_scores(query) ) # 유사도 있는 비동기 개체호출 

# print(query +"  >> 답변에 사용할 문장 문장 검색 ")

# print("-"*100)

# for doc, score in docs:
#     print(f"{score}\t{doc.page_content}")

# print("="*00)

from langchain.chat_models import ChatOpenAI

llm_model = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0) 
messages=[
    {"role": "system", "content": "너는 문화관련 내용으로 기사을 쓰는 기자야 항상 답변 전에 안녕하세요 하고 인사를 해야해"},
    {"role": "user", "content": "이 내용들을 바탕으로 기사를 작성해줘"},
    {"role": "assistant", "content": "제주특별자치도 도립미술관(관장 이나연)은 온라인제주도립미술관 웹 플랫폼을 활용한 온라인전시 기획안을 접수한다.\n‘온라인제주도립미술관 전시 공모전’은 문화체육관광부의 2021 스마트 박물관·미술관 구축 자원 사업으로 구축된 온라인제주도립미술관 웹 플랫폼에 일반인이 참여하는 첫 공모전이다.\n공모 자격은 제주도에 주소지를 둔 만 19세(2004년 1월 1일 이전 출생) 이상 도민으로, 제주도립미술관의 소장품을 대상으로 자유롭게 주제를 선정해 전시를 기획하면 된다.\n관련 공고는 6월 12일까지이며, 전시 기획안은 오는 6월 13~19일 온라인제주도립미술관 웹 플랫폼에서만 접수가능하다.\n심사 기준은 주제의 참신성, 미술 기초지식, 작품 구성에 대한 타당성 등을 종합 평가해 선정한 최종 1개의 기획안을 6월 24일 발표할 예정이다.\n선정된 기획안은 2022년 7월 중 온라인제주도립미술관 웹 플랫폼의 가상전시공간에서 구현할 예정이다.\n자세한 사항은 제주도립미술관 누리집과 온라인제주도립미술관 웹 플랫폼을 참고하거나 온라인제주도립미술관 전시 공모 담당자(☎064-710-4274)에게 문의하면 된다."},
    {"role": "assistant", "content":"제주특별자치도 도립미술관(관장 이나연)은 19일부터 9월 18일까지 온라인제주도립미술관 공모전 수상 전시기획인 ‘삶과 사람 사이’를 가상 전시 공간에서 진행한다고 밝혔다.\n도립미술관은 일반인과 협업해 새로운 개념의 전시 콘텐츠를 통해 소통의 창구를 확장한다는 취지로 지난 5월부터 제주도민을 대상으로 제주도립미술관 소장품으로 구성하는 공모 전시 기획안을 접수받았다. \n이후 6월 24일 최종 선정된 일반인과 여러 차례 협의해 이번 전시 과정을 지원하게 됐다. \n이번 전시는 온라인제주도립미술관의 두 번째 전시로, 코로나19 전후를 비교해 삶의 모습이 어떻게 변화됐는지 두 개의 섹션으로 구분해 인물의 다양한 모습을 표현한다.  \n‘이전의 날들’에서는 사람들이 함께 어울리는 평범한 일상의 모습을 통해 그동안 누려온 당연한 것들이 얼마나 소중한 것이었는지 느끼게 해주는 작품들을 보여준다.\n‘이후의 날들’은 감염병 확산에 대한 두려움으로 다수의 사람들을 경계하며 혼란스러워하는 사람들과 개인의 고립된 모습을 담은 작품으로 구성했다.\n이나연 도립미술관장은 “이번 공모전을 시작으로 일반인들이 직접 전시기획에 참여할 수 있는 기회를 제공하고, 온라인제주도립미술관 웹 플랫폼의 활용방안을 다각화할 수 있도록 지속적으로 노력하겠다”고 말했다\n한편, 온라인제주도립미술관 전시는 컴퓨터(PC) 및 모바일 웹 플랫폼에서 제공되며 검색창에 온라인제주도립미술관(http://www.onlinejmoa.or.kr)을 입력하거나 제주도립미술관 누리집(http://jmoa.jeju.go.kr)에서 온라인제주도립미술관에 접속하면 된다."},
    {"role": "assistant", "content": "3D 가상전시실, 360°VR 야외조각공원, 고해상도 이미지 확대 등 비대면 서비스 구현\n제주특별자치도 도립미술관(관장 이나연)이 4차산업혁명 시대에 정보통신기술(ICT)을 활용한 온라인제주도립미술관 웹 플랫폼(www.onlinejmoa.or.kr)을 구축하고, 17일부터 가상 전시 ‘풍경을 잇다’전을 개최한다고 16일 밝혔다.\n온라인제주도립미술관은 문화체육관광부의 2021년 스마트 공립박물관·미술관 구축 지원 사업 공모에 선정돼 시·공간 제한 없는 비대면 관람 환경을 제공하기 위해 구축됐다.\n​3D 스캐닝 및 모델링 기술로 구현한 가상전시실은 실제 제주도립미술관 기획전시실2를 방문해 작품을 관람하는 것처럼 입체적인 공간감을 느끼게 하고, ​360°VR 메타포트 연계 기술로 촬영한 제주도립미술관의 야외조각공원에서는 사계절의 아름다움을 모두 표현할 예정이다.\n​또한, 고해상도 작품 이미지 확대 서비스를 통해 작품의 세밀한 붓질과 생생한 색감을 감상할 수도 있다.\n온라인제주도립미술관 웹 플랫폼은 제주도립미술관 홈페이지에서 온라인제주도립미술관을 클릭해 접속하면 된다.\n​이나연 도립미술관장은 “비대면 콘텐츠가 요구되는 시대에 발맞춰 가상 전시를 관람할 수 있는 기회를 마련했다”며 “참여형 온라인공모 전시를 통해 일반인이 직접 큐레이팅할 수 있는 기획전을 열고 관람객과 한층 더 가까운 미술관이 되도록 노력할 것”이라고 말했다.\n자세한 문의는 제주도립미술관(064-710-4274)으로 하면 된다."},
    {"role": "assistant", "content": "제주특별자치도 도립미술관이 4차산업혁명 시대에 정보통신기술을 활용한 온라인제주도립미술관 웹 플랫폼을 구축하고 17일부터 가상 전시 ‘풍경을 잇다’전을 선보인다고 밝혔다.\n온라인제주도립미술관은 문화체육관광부의 2021년 스마트 공립박물관·미술관 구축 지원 사업 공모에 선정돼 시·공간 제한 없는 비대면 관람 환경을 제공하기 위해 구축됐다.\n3D 스캐닝 및 모델링 기술로 구현한 가상전시실은 실제 제주도립미술관 기획전시실2를 방문해 작품을 관람하는 것처럼 입체적인 공간감을 느끼게 한다.\n360°VR 메타포트 연계 기술로 촬영한 제주도립미술관의 야외조각공원에서는 사계절의 아름다움을 모두 표현할 예정이다.\n또한, 고해상도 작품 이미지 확대 서비스를 통해 작품의 세밀한 붓질과 생생한 색감을 감상할 수도 있다.\n온라인제주도립미술관 첫 전시로 마련된 소장품 기획전 ‘풍경을 잇다’는 제주도립미술관 소장품 중 제주 풍경을 그린 26명의 작가의 작품 29점으로 구성했다.\n전시에서는 제주의 풍경화를 동서남북으로 구분해 4개의 섹션으로 소개한다.\n전시를 통해 제주를 랜선 여행하는 즐거움과 예술가의 예민한 감성으로 바라본 제주의 아름다움을 함께 느끼기를 기대한다.\n가상 전시는 오는 7월 17일까지 62일간 이어진다.\n온라인제주도립미술관 웹 플랫폼은 제주도립미술관 홈페이지에서 온라인제주도립미술관을 클릭해 접속하면 된다."},
    {"role": "assistant", "content": "제주특별자치도립미술관(관장 이나연)이 정보통신기술(ICT)을 활용한 ‘온라인제주도립미술관’ 웹 플랫폼을 구축, 지난 17일부터 오는 7월 17일까지 가상 전시 ‘풍경을 잇다’전을 열고 있다.\n‘온라인제주도립미술관’ 웹 플랫폼은 제주도립미술관 홈페이지에서 온라인제주도립미술관을 클릭해 접속하면 된다.\n제주도립미술관이 3D 스캐닝 및 모델링 기술로 구현한 가상전시실은 실제 제주도립미술관 기획전시실을 방문해 작품을 관람하는 것처럼 입체적인 공간감을 느끼게 한다.\n특히 고해상도 작품 이미지 확대 서비스를 통해 작품의 세밀한 붓질과 생생한 색감을 감상할 수 있다.\n온라인제주도립미술관 첫 전시로 마련된 소장품 기획전 '풍경을 잇다'는 제주도립미술관 소장품 중 제주 풍경을 그린 26명의 작가의 작품 29점으로 구성됐다.\n참여 작가는 고순철, 공성훈, 강요배, 김병화, 김성오, 김인지, 김천희, 문인환, 문행섭, 박성배, 변시지, 부현일, 손장섭, 양근석, 양창보, 오승익, 유창훈, 윤재우, 이동근, 이옥문, 이옥구, 이창희, 임갑재, 임현자, 한중옥, 홍종명 등이다.\n전시에서는 제주의 풍경화를 동서남북으로 구분해 4개의 섹션으로 소개되고 있다.\n한편 ‘온라인제주도립미술관’은 문화체육관광부의 2021년 스마트 공립박물관·미술간 구축 지원 사업 공모에 선정돼 구축됐다."},
    {"role": "assistant", "content":"제주도 도립미술관은 17일부터 가상 전시 ‘풍경을 잇다’를 선보인다고 16일 밝혔다.이번 가상 전시는 지난해 문화체육관광부의 스마트공립미술관 구축 지원 사업 공모에 선정돼 마련된 것으로, 온라인제주도립미술관 웹 플랫폼에서 만나볼 수 있다.도립미술관은 3D 스캐닝 및 모델링 기술로 가상전시실을 구현해 실제 제주도립미술관 기획전시실2를 그대로 온라인 상에 담아냈다.여기에 360도 VR 메타포트 연계 기술로 촬영한 콘텐츠를 담아 냈으며, 도립미술관의 야외 조각공원의 경우 사계절의 아름다움을 모두 만나볼 수 있다.미술관 관계자는 “고해상도 작품 이미지 확대 서비스를 통해 작품의 세밀한 붓질과 생생한 색감을 감상할 수 있다”고 전했다.도립미술관은 첫 전시로 소장품 기획전 ‘풍경을 잇다’를 진행, 소장품 중 제주 풍경을 그린 작품 29점을 선보인다.전시는 제주의 풍경화를 동서남북으로 구분해 4개의 섹션으로 소개한다. 전시는 오는 7월 17일까지 이어진다.미술관 관계자는 “비대면 콘텐츠가 요구되는 시대에 발맞춰 가상 전시를 관람할 수 있는 기회를 마련했다”며 “참여형 온라인공모 전시를 통해 일반인이 직접 큐레이팅할 수 있는 기획전도 준비 중”이라고 전했다."}
]

chain = load_qa_chain(llm_model, verbose=False)


for i in range(0,3):
    query = "이 내용들을 바탕으로 행사의 기간, 행사의 주제, 행사의 내용, 행사 장소등을 포함한 800자 내외의 기사를 작성해줘, "
    print('\n')
    docs = index.similarity_search(query)
    res = chain.run(input_documents=docs, question=query)
    print(f'{i+1}번째 기사입니다 \n')
    print( query,res,end='\n\n')

