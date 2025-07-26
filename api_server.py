from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import traceback
import os
import sys
from dotenv import load_dotenv

# 确保 src 路径可导入
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.hipporag.MultiTenantHippoRAGManager import MultiTenantHippoRAGManager

app = FastAPI()
# 加载环境变量文件
load_dotenv()
# 初始化多租户管理器
BASE_SAVE_DIR = os.environ.get('HIPPORAG_SAVE_DIR', None)
if BASE_SAVE_DIR is None:
    raise ValueError('HIPPORAG_SAVE_DIR is not set')
# 创建目录
os.makedirs(BASE_SAVE_DIR, exist_ok=True)
manager = MultiTenantHippoRAGManager(base_save_dir=BASE_SAVE_DIR)

class IndexRequest(BaseModel):
    tenant_id: str
    docs: List[str]

class IndexResponse(BaseModel):
    code: int
    msg: str
    data: Dict[str, Any] = {}

class RetrieveRequest(BaseModel):
    tenant_id: str
    querys: List[str]

class RetrieveResponse(BaseModel):
    code: int
    msg: str
    data: Dict[str, Any]

@app.post('/index', response_model=IndexResponse)
def index_api(req: IndexRequest):
    try:
        rag = manager.get_tenant_config(req.tenant_id)
        rag.index(req.docs)
        return IndexResponse(code=0, msg='索引成功', data={})
    except Exception as e:
        traceback.print_exc()
        return IndexResponse(code=1001, msg=f'索引失败: {str(e)}', data={})

@app.post('/retrieve', response_model=RetrieveResponse)
def retrieve_api(req: RetrieveRequest):
    try:
        rag = manager.get_tenant_config(req.tenant_id)
        results = rag.retrieve(req.querys)
        # 只返回 docs 字段
        docs_list = []
        for r in results:
            # QuerySolution 结构: question, docs, doc_scores
            docs_list.append(r.docs if hasattr(r, 'docs') else [])
        return RetrieveResponse(code=0, msg='检索成功', data={'docs': docs_list})
    except Exception as e:
        traceback.print_exc()
        return RetrieveResponse(code=1002, msg=f'检索失败: {str(e)}', data={'docs': []})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=6200, reload=False) 