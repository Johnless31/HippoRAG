from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import traceback
import os
import sys
import shutil
from dotenv import load_dotenv

# 确保 src 路径可导入
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.hipporag.MultiTenantHippoRAGManager import MultiTenantHippoRAGManager

app = FastAPI()
# 加载环境变量文件
load_dotenv()

# 设置默认的API密钥（用于本地模型）
if os.getenv('OPENAI_API_KEY') is None:
    os.environ['OPENAI_API_KEY'] = 'sk-687b04566be54aefb99718c096e1acde'
    print("🔑 设置默认API密钥: sk-687b04566be54aefb99718c096e1acde")

# 初始化多租户管理器
BASE_SAVE_DIR = os.environ.get('HIPPORAG_SAVE_DIR', None)
if BASE_SAVE_DIR is None:
    # 如果没有设置环境变量，使用当前目录下的data文件夹
    BASE_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    print(f"⚠️ HIPPORAG_SAVE_DIR 环境变量未设置，使用默认路径: {BASE_SAVE_DIR}")

# 创建目录
os.makedirs(BASE_SAVE_DIR, exist_ok=True)
print(f"📁 使用数据保存目录: {BASE_SAVE_DIR}")

# 检查是否存在旧的outputs目录，如果存在则迁移数据
old_outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
if os.path.exists(old_outputs_dir):
    print(f"🔄 检测到旧的outputs目录，正在迁移数据...")
    # 这里可以添加数据迁移逻辑，暂时只是提示
    print(f"💡 请手动将 {old_outputs_dir} 中的数据迁移到 {BASE_SAVE_DIR}")

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

class DeleteRequest(BaseModel):
    tenant_id: str

class DeleteResponse(BaseModel):
    code: int
    msg: str
    data: Dict[str, Any] = {}

@app.post('/index', response_model=IndexResponse)
def index_api(req: IndexRequest):
    """
    索引接口 - 为指定租户建立文档索引
    
    Args:
        req: 包含租户ID和文档列表的请求
        
    Returns:
        索引操作结果
    """
    try:
        rag = manager.get_tenant_config(req.tenant_id)
        rag.index(req.docs)
        return IndexResponse(code=0, msg='索引成功', data={})
    except Exception as e:
        traceback.print_exc()
        return IndexResponse(code=1001, msg=f'索引失败: {str(e)}', data={})

@app.post('/retrieve', response_model=RetrieveResponse)
def retrieve_api(req: RetrieveRequest):
    """
    检索接口 - 从指定租户的知识图谱中检索相关文档
    
    Args:
        req: 包含租户ID和查询列表的请求
        
    Returns:
        检索结果，包含相关文档列表
    """
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

@app.delete('/delete', response_model=DeleteResponse)
def delete_api(req: DeleteRequest):
    """
    删除接口 - 删除指定租户的所有数据和实例
    
    Args:
        req: 包含租户ID的请求
        
    Returns:
        删除操作结果
    """
    try:
        tenant_id = req.tenant_id
        
        print(f"🗑️ 开始删除租户 {tenant_id} 的数据...")
        
        # 1. 清理内存中的实例
        if tenant_id in manager._tenant_instances:
            manager._remove_tenant_instance(tenant_id)
            print(f"✅ 已清理租户 {tenant_id} 的内存实例")
        
        # 2. 从配置中移除
        if tenant_id in manager.tenant_configs:
            del manager.tenant_configs[tenant_id]
            manager.save_tenant_configs()
            print(f"✅ 已从配置中移除租户 {tenant_id}")
        
        # 3. 删除文件系统中的数据
        tenant_dir = os.path.join(BASE_SAVE_DIR, tenant_id)
        if os.path.exists(tenant_dir):
            try:
                shutil.rmtree(tenant_dir)
                print(f"✅ 已删除租户 {tenant_id} 的文件系统数据: {tenant_dir}")
            except Exception as e:
                print(f"⚠️ 删除文件系统数据时出错: {e}")
                return DeleteResponse(
                    code=1003, 
                    msg=f'删除租户 {tenant_id} 文件失败: {str(e)}', 
                    data={}
                )
        else:
            print(f"ℹ️ 租户 {tenant_id} 的文件目录不存在: {tenant_dir}")
        
        # 4. 清理访问时间记录
        if tenant_id in manager._access_times:
            del manager._access_times[tenant_id]
        
        return DeleteResponse(
            code=0, 
            msg=f'租户 {tenant_id} 删除成功', 
            data={
                'tenant_id': tenant_id,
                'deleted_dir': tenant_dir
            }
        )
        
    except Exception as e:
        traceback.print_exc()
        return DeleteResponse(
            code=1003, 
            msg=f'删除租户 {req.tenant_id} 失败: {str(e)}', 
            data={}
        )

@app.get('/tenants')
def list_tenants():
    """
    列出所有租户接口
    
    Returns:
        所有租户的列表和状态信息
    """
    try:
        active_tenants = manager.list_active_tenants()
        all_tenants = list(manager.tenant_configs.keys())
        
        tenant_info = []
        for tenant_id in all_tenants:
            tenant_dir = os.path.join(BASE_SAVE_DIR, tenant_id)
            tenant_info.append({
                'tenant_id': tenant_id,
                'active': tenant_id in active_tenants,
                'config_exists': tenant_id in manager.tenant_configs,
                'files_exist': os.path.exists(tenant_dir),
                'last_access': manager._access_times.get(tenant_id, None)
            })
        
        return {
            'code': 0,
            'msg': '获取租户列表成功',
            'data': {
                'total_tenants': len(all_tenants),
                'active_tenants': len(active_tenants),
                'tenants': tenant_info
            }
        }
        
    except Exception as e:
        traceback.print_exc()
        return {
            'code': 1004,
            'msg': f'获取租户列表失败: {str(e)}',
            'data': {}
        }

if __name__ == '__main__':
    import uvicorn
    print("🚀 启动HippoRAG API服务器...")
    print(f"📍 服务器地址: http://0.0.0.0:6200")
    print(f"📁 数据保存目录: {BASE_SAVE_DIR}")
    print("📚 可用接口:")
    print("   - POST /index: 建立文档索引")
    print("   - POST /retrieve: 检索相关文档")
    print("   - DELETE /delete: 删除租户数据")
    print("   - GET /tenants: 列出所有租户")
    uvicorn.run("api_server:app", host="0.0.0.0", port=6200, reload=False) 