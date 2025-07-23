#!/usr/bin/env python3
"""
基于ChromaDB的HippoRAG图谱可视化
从ChromaDB存储中加载图谱数据并生成交互式可视化
"""

import os
import sys
import json
import pickle
import networkx as nx
from pyvis.network import Network
from collections import defaultdict, Counter
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

# 添加项目路径
sys.path.append('src')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import igraph as ig
except ImportError:
    logger.error("❌ 需要安装igraph: pip install igraph")
    sys.exit(1)

try:
    import chromadb
    from src.hipporag.chroma_store import ChromaStore
    CHROMA_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ ChromaDB不可用，将仅支持传统embedding store")
    ChromaStore = None
    CHROMA_AVAILABLE = False

class EmbeddingStoreAdapter:
    """适配器类，用于兼容传统embedding store数据格式"""
    
    def __init__(self, data: Dict):
        self.data = data
    
    def count(self) -> int:
        return len(self.data)
    
    def get_all_ids(self) -> List[str]:
        return list(self.data.keys())
    
    def get_all_texts(self) -> List[str]:
        return [item['content'] for item in self.data.values()]
    
    def get_all_id_to_rows(self) -> Dict[str, Dict[str, str]]:
        return self.data
    
    def get_row(self, hash_id: str) -> Dict[str, str]:
        if hash_id not in self.data:
            raise KeyError(f"Hash ID {hash_id} not found")
        return self.data[hash_id]
    
    def get_rows(self, hash_ids: List[str]) -> Dict[str, Dict[str, str]]:
        return {hash_id: self.data[hash_id] for hash_id in hash_ids if hash_id in self.data}

class ChromaGraphVisualizer:
    """基于ChromaDB的HippoRAG图谱可视化器"""
    
    def __init__(self, working_dir: str, namespace: str = "default"):
        """
        初始化可视化器
        
        Args:
            working_dir: HippoRAG工作目录
            namespace: 命名空间，用于区分不同的实验
        """
        self.working_dir = working_dir
        self.namespace = namespace
        
        # ChromaDB路径
        self.chunk_db_path = os.path.join(working_dir, "chroma_chunk_embeddings")
        self.entity_db_path = os.path.join(working_dir, "chroma_entity_embeddings")
        self.fact_db_path = os.path.join(working_dir, "chroma_fact_embeddings")
        
        # 图谱文件路径
        self.graph_pickle_path = os.path.join(working_dir, "graph.pickle")
        self.openie_results_path = None
        
        # 初始化ChromaStore
        self.chunk_store = None
        self.entity_store = None
        self.fact_store = None
        
        # 数据容器
        self.igraph_obj = None
        self.openie_data = None
        
        logger.info(f"ChromaGraphVisualizer initialized for {working_dir}")
    
    def load_chroma_stores(self):
        """加载ChromaDB存储"""
        logger.info("🔄 加载ChromaDB存储...")
        
        # 检查是否有ChromaDB数据
        chroma_exists = (
            os.path.exists(self.chunk_db_path) and 
            os.path.exists(self.entity_db_path) and 
            os.path.exists(self.fact_db_path)
        )
        
        if not chroma_exists or not CHROMA_AVAILABLE:
            logger.info("⚠️ 未找到ChromaDB数据或ChromaDB不可用，尝试从传统embedding store转换...")
            return self._convert_from_embedding_store()
        
        try:
            # 初始化ChromaStore（不需要embedding模型，只用于数据访问）
            self.chunk_store = ChromaStore(None, self.chunk_db_path, 32, 'chunk')
            self.entity_store = ChromaStore(None, self.entity_db_path, 32, 'entity')
            self.fact_store = ChromaStore(None, self.fact_db_path, 32, 'fact')
            
            # 检查数据
            chunk_count = self.chunk_store.count()
            entity_count = self.entity_store.count()
            fact_count = self.fact_store.count()
            
            logger.info(f"✅ 成功加载ChromaDB存储:")
            logger.info(f"   - 文档块: {chunk_count}")
            logger.info(f"   - 实体: {entity_count}")
            logger.info(f"   - 事实: {fact_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载ChromaDB存储失败: {e}")
            return False
    
    def _convert_from_embedding_store(self):
        """从传统embedding store转换数据"""
        logger.info("🔄 从传统embedding store加载数据...")
        
        # 检查传统embedding store路径
        traditional_chunk_path = os.path.join(self.working_dir, "chunk_embeddings")
        traditional_entity_path = os.path.join(self.working_dir, "entity_embeddings")
        traditional_fact_path = os.path.join(self.working_dir, "fact_embeddings")
        
        if not (os.path.exists(traditional_chunk_path) and 
                os.path.exists(traditional_entity_path) and 
                os.path.exists(traditional_fact_path)):
            logger.error("❌ 未找到传统embedding store数据")
            return False
        
        try:
            # 从传统embedding store加载数据
            self.chunk_data = self._load_traditional_embedding_store(traditional_chunk_path, 'chunk')
            self.entity_data = self._load_traditional_embedding_store(traditional_entity_path, 'entity')
            self.fact_data = self._load_traditional_embedding_store(traditional_fact_path, 'fact')
            
            logger.info(f"✅ 成功从传统embedding store加载数据:")
            logger.info(f"   - 文档块: {len(self.chunk_data)}")
            logger.info(f"   - 实体: {len(self.entity_data)}")
            logger.info(f"   - 事实: {len(self.fact_data)}")
            
            # 创建模拟的ChromaStore对象
            self.chunk_store = EmbeddingStoreAdapter(self.chunk_data)
            self.entity_store = EmbeddingStoreAdapter(self.entity_data)
            self.fact_store = EmbeddingStoreAdapter(self.fact_data)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 从传统embedding store加载数据失败: {e}")
            return False
    
    def _load_traditional_embedding_store(self, db_path: str, namespace: str) -> Dict:
        """加载传统embedding store数据"""
        parquet_file = os.path.join(db_path, f"vdb_{namespace}.parquet")
        if not os.path.exists(parquet_file):
            logger.error(f"❌ 找不到文件: {parquet_file}")
            return {}
        
        df = pd.read_parquet(parquet_file)
        
        # 转换为字典格式
        data = {}
        for _, row in df.iterrows():
            data[row['hash_id']] = {
                'hash_id': row['hash_id'],
                'content': row['content']
            }
        
        return data
    
    def load_graph(self):
        """加载图谱结构"""
        logger.info("🔄 加载图谱结构...")
        
        if not os.path.exists(self.graph_pickle_path):
            logger.error(f"❌ 找不到图谱文件: {self.graph_pickle_path}")
            return False
        
        try:
            with open(self.graph_pickle_path, 'rb') as f:
                self.igraph_obj = pickle.load(f)
            
            logger.info(f"✅ 成功加载图谱: {self.igraph_obj.summary()}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载图谱失败: {e}")
            return False
    
    def load_openie_data(self):
        """加载OpenIE数据（如果存在）"""
        # 查找OpenIE结果文件
        possible_files = [
            os.path.join(self.working_dir, "openie_results.json"),
            os.path.join(os.path.dirname(self.working_dir), "openie_results_ner_deepseek-v3.json"),
            os.path.join(os.path.dirname(self.working_dir), "openie_results.json"),
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                self.openie_results_path = file_path
                break
        
        if self.openie_results_path is None:
            logger.warning("⚠️ 未找到OpenIE结果文件，将使用基本的图谱信息")
            return False
        
        try:
            with open(self.openie_results_path, 'r', encoding='utf-8') as f:
                self.openie_data = json.load(f)
            
            logger.info(f"✅ 成功加载OpenIE数据: {len(self.openie_data.get('docs', []))} 个文档")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 加载OpenIE数据失败: {e}")
            return False
    
    def extract_semantic_info(self):
        """从OpenIE数据中提取语义信息"""
        if not self.openie_data:
            return {}, {}, {}, Counter(), {}
        
        entity_contexts = defaultdict(list)
        entity_relations = defaultdict(list)
        entity_types = defaultdict(set)
        relation_mappings = defaultdict(list)
        entity_frequencies = Counter()
        
        for doc in self.openie_data.get('docs', []):
            passage = doc.get('passage', '')
            entities = doc.get('extracted_entities', [])
            triples = doc.get('extracted_triples', [])
            
            # 收集所有实体
            all_entities = set(entities)
            for triple in triples:
                if len(triple) >= 3:
                    subject, relation, obj = triple[0], triple[1], triple[2]
                    all_entities.add(subject)
                    all_entities.add(obj)
            
            # 记录上下文和频率
            for entity in all_entities:
                entity_contexts[entity].append(passage)
                entity_frequencies[entity] += 1
            
            # 记录关系
            for triple in triples:
                if len(triple) >= 3:
                    subject, relation, obj = triple[0], triple[1], triple[2]
                    relation_str = f"{subject} → {relation} → {obj}"
                    entity_relations[subject].append(relation_str)
                    entity_relations[obj].append(relation_str)
                    relation_mappings[(subject, obj)].append((relation, passage))
                    relation_mappings[(obj, subject)].append((relation, passage))
        
        # 推断实体类型
        for entity in entity_frequencies:
            relations = entity_relations[entity]
            inferred_types = self._infer_entity_type(entity, relations)
            entity_types[entity].update(inferred_types)
        
        return entity_contexts, entity_relations, entity_types, entity_frequencies, relation_mappings
    
    def _infer_entity_type(self, entity_name: str, relations: List[str]) -> List[str]:
        """从关系中推断实体类型"""
        entity_types = set()
        
        for relation_str in relations:
            if "is a" in relation_str:
                parts = relation_str.split(" → ")
                if len(parts) == 3 and parts[1] == "is a" and parts[0] == entity_name:
                    entity_types.add(parts[2])
            elif "born in" in relation_str or "birthplace" in relation_str:
                if entity_name in relation_str.split(" → ")[0]:
                    entity_types.add("Person")
                else:
                    entity_types.add("Place")
        
        # 根据实体名称推断
        if "County" in entity_name:
            entity_types.add("Administrative Region")
        elif entity_name.lower() in ["cinderella", "prince"]:
            entity_types.add("Character")
        elif entity_name.lower() in ["kingdom", "royal ball"]:
            entity_types.add("Place/Event")
        elif "slipper" in entity_name.lower():
            entity_types.add("Object")
        elif entity_name.lower() == "politician":
            entity_types.add("Profession/Role")
        
        return list(entity_types)
    
    def build_networkx_graph(self):
        """构建NetworkX图"""
        logger.info("🔄 构建NetworkX图...")
        
        if not self.igraph_obj:
            logger.error("❌ 图谱未加载")
            return None
        
        # 提取语义信息
        entity_contexts, entity_relations, entity_types, entity_frequencies, relation_mappings = self.extract_semantic_info()
        
        # 创建NetworkX图
        G = nx.Graph()
        
        # 获取ChromaDB数据
        chunk_data = self.chunk_store.get_all_id_to_rows()
        entity_data = self.entity_store.get_all_id_to_rows()
        fact_data = self.fact_store.get_all_id_to_rows()
        
        # 诊断信息
        matched_entities = []
        missing_entities = []
        
        # 添加节点
        for i in range(self.igraph_obj.vcount()):
            node_attrs = {}
            for attr in self.igraph_obj.vertex_attributes():
                node_attrs[attr] = self.igraph_obj.vs[i][attr]
            
            node_id = node_attrs.get('name', f'node_{i}')
            node_content = node_attrs.get('content', '').lower()
            
            # 融合语义信息
            if node_id.startswith('entity-'):
                # 查找对应的实体
                matching_entity = None
                for entity in entity_frequencies:
                    if entity.lower() == node_content:
                        matching_entity = entity
                        break
                
                if matching_entity:
                    matched_entities.append((node_id, matching_entity, node_content))
                    node_attrs['semantic_name'] = matching_entity
                    node_attrs['contexts'] = entity_contexts[matching_entity]
                    node_attrs['relations'] = entity_relations[matching_entity]
                    node_attrs['inferred_types'] = list(entity_types[matching_entity])
                    node_attrs['frequency'] = entity_frequencies[matching_entity]
                else:
                    missing_entities.append((node_id, node_content))
                    node_attrs['semantic_name'] = node_content if node_content else "未知实体"
                    node_attrs['contexts'] = []
                    node_attrs['relations'] = []
                    node_attrs['inferred_types'] = []
                    node_attrs['frequency'] = 0
            
            elif node_id.startswith('chunk-'):
                # 查找对应的文档块
                if node_id in chunk_data:
                    chunk_info = chunk_data[node_id]
                    node_attrs['full_content'] = chunk_info['content']
                
            G.add_node(node_id, **node_attrs)
        
        # 添加边
        for i in range(self.igraph_obj.ecount()):
            edge = self.igraph_obj.es[i]
            source_idx = edge.source
            target_idx = edge.target
            
            source_name = self.igraph_obj.vs[source_idx]['name']
            target_name = self.igraph_obj.vs[target_idx]['name']
            
            edge_attrs = {}
            for attr in self.igraph_obj.edge_attributes():
                edge_attrs[attr] = edge[attr]
            
            # 融合语义关系信息
            source_content = self.igraph_obj.vs[source_idx]['content'] if 'content' in self.igraph_obj.vs[source_idx].attributes() else ''
            target_content = self.igraph_obj.vs[target_idx]['content'] if 'content' in self.igraph_obj.vs[target_idx].attributes() else ''
            
            # 查找语义关系
            semantic_relations = []
            if (source_content, target_content) in relation_mappings:
                semantic_relations.extend(relation_mappings[(source_content, target_content)])
            if (target_content, source_content) in relation_mappings:
                semantic_relations.extend(relation_mappings[(target_content, source_content)])
            
            edge_attrs['semantic_relations'] = semantic_relations
            
            G.add_edge(source_name, target_name, **edge_attrs)
        
        # 打印诊断信息
        logger.info(f"🔍 图谱构建完成:")
        logger.info(f"   - 节点总数: {G.number_of_nodes()}")
        logger.info(f"   - 边总数: {G.number_of_edges()}")
        logger.info(f"   - 成功匹配实体: {len(matched_entities)}")
        logger.info(f"   - 缺失实体: {len(missing_entities)}")
        
        # 返回图和诊断信息
        diagnostics = (matched_entities, missing_entities, [])
        return G, diagnostics
    
    def get_entity_color(self, entity_content: str, inferred_types: List[str]) -> str:
        """根据实体内容和类型确定颜色"""
        content_lower = entity_content.lower()
        
        if 'politician' in content_lower or 'Profession/Role' in inferred_types:
            return "#96CEB4"  # 绿色 - 职业
        elif any(name in content_lower for name in ['cinderella', 'prince', 'person']):
            return "#FF6B6B"  # 红色 - 人物
        elif any(place in content_lower for place in ['county', 'kingdom', 'place']) or any(t in inferred_types for t in ['Place', 'Administrative Region']):
            return "#4ECDC4"  # 青色 - 地点
        elif 'slipper' in content_lower or 'Object' in inferred_types:
            return "#45B7D1"  # 蓝色 - 物品
        elif 'event' in content_lower or any(t in inferred_types for t in ['Place/Event', 'Event']):
            return "#DDA0DD"  # 紫色 - 事件
        else:
            return "#FFEAA7"  # 黄色 - 其他
    
    def create_node_description(self, node_id: str, node_data: Dict, degrees: Dict) -> str:
        """创建节点描述"""
        desc_parts = []
        
        if node_id.startswith('entity-'):
            content = node_data.get('content', 'Unknown')
            semantic_name = node_data.get('semantic_name', content)
            contexts = node_data.get('contexts', [])
            relations = node_data.get('relations', [])
            inferred_types = node_data.get('inferred_types', [])
            frequency = node_data.get('frequency', 0)
            
            desc_parts.append(f"🎯 实体: {semantic_name}")
            if inferred_types:
                desc_parts.append(f"🏷️ 类型: {', '.join(inferred_types)}")
            desc_parts.append(f"🔗 图谱度: {degrees[node_id]}")
            desc_parts.append(f"📊 语义频率: {frequency}")
            desc_parts.append(f"📋 节点ID: {node_id}")
            desc_parts.append(f"🔑 Hash: {node_data.get('hash_id', 'N/A')}")
            
            if contexts:
                desc_parts.append(f"📖 上下文:")
                for i, context in enumerate(set(contexts[:3]), 1):
                    desc_parts.append(f"  {i}. {context[:100]}...")
            
            if relations:
                desc_parts.append(f"🔗 语义关系:")
                unique_relations = list(set(relations[:5]))
                for i, relation in enumerate(unique_relations, 1):
                    desc_parts.append(f"  {i}. {relation}")
        
        elif node_id.startswith('chunk-'):
            content = node_data.get('content', 'Unknown')
            full_content = node_data.get('full_content', content)
            desc_parts.append(f"📄 文档块: {full_content[:100]}...")
            desc_parts.append(f"🔗 图谱度: {degrees[node_id]}")
            desc_parts.append(f"📋 节点ID: {node_id}")
            desc_parts.append(f"🔑 Hash: {node_data.get('hash_id', 'N/A')}")
        
        return "\n".join(desc_parts)
    
    def create_edge_description(self, source: str, target: str, edge_data: Dict) -> str:
        """创建边描述"""
        weight = edge_data.get('weight', 1.0)
        semantic_relations = edge_data.get('semantic_relations', [])
        
        desc_parts = []
        desc_parts.append(f"🔗 连接: {source} ↔ {target}")
        desc_parts.append(f"⚖️ 权重: {weight:.3f}")
        
        if semantic_relations:
            desc_parts.append(f"📝 语义关系:")
            for i, (relation, context) in enumerate(semantic_relations[:3], 1):
                desc_parts.append(f"  {i}. {relation}")
                desc_parts.append(f"     来源: {context[:50]}...")
        else:
            desc_parts.append(f"📝 语义关系: 基于嵌入相似度连接")
        
        return "\n".join(desc_parts)
    
    def visualize_graph(self, G: nx.Graph, output_file: str, diagnostics: Tuple = None):
        """可视化图谱"""
        logger.info("🎨 生成可视化...")
        
        # 计算度分布
        degrees = dict(G.degree())
        
        # 分析节点类型
        entity_nodes = [n for n in G.nodes() if n.startswith('entity-')]
        chunk_nodes = [n for n in G.nodes() if n.startswith('chunk-')]
        
        # 创建pyvis网络
        net = Network(height="900px", width="100%", bgcolor="#f8f9fa", font_color="black")
        
        # 设置物理效果
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 300},
            "barnesHut": {
              "gravitationalConstant": -15000,
              "centralGravity": 0.15,
              "springLength": 120,
              "springConstant": 0.03,
              "damping": 0.09
            }
          },
          "interaction": {
            "hover": true,
            "selectConnectedEdges": true,
            "tooltipDelay": 300
          },
          "manipulation": {
            "enabled": false
          }
        }
        """)
        
        # 添加节点
        for node_id, node_data in G.nodes(data=True):
            node_degree = degrees[node_id]
            frequency = node_data.get('frequency', 0)
            base_size = 18
            size = base_size + node_degree * 3 + frequency * 2
            
            if node_id.startswith('entity-'):
                content = node_data.get('semantic_name', node_data.get('content', ''))
                inferred_types = node_data.get('inferred_types', [])
                
                if content == "未知实体" or content == "":
                    color = "#FF0000"  # 红色 - 缺失实体
                    label = "❌未知"
                else:
                    color = self.get_entity_color(content, inferred_types)
                    label = content[:15] + "..." if len(content) > 15 else content
            elif node_id.startswith('chunk-'):
                color = "#000000"  # 黑色 - 文档
                label = f"Doc-{node_id[-8:]}"
            else:
                color = "#B2B2B2"  # 灰色 - 未知
                label = node_id
            
            # 创建描述
            description = self.create_node_description(node_id, node_data, degrees)
            
            net.add_node(node_id,
                         label=label,
                         size=size,
                         color=color,
                         title=description,
                         font={'size': 14, 'color': 'black'})
        
        # 添加边
        for source, target, edge_data in G.edges(data=True):
            weight = edge_data.get('weight', 1.0)
            semantic_relations = edge_data.get('semantic_relations', [])
            
            width = max(2, int(weight * 4))
            
            if semantic_relations:
                if weight >= 1.5:
                    color = "#E74C3C"  # 红色 - 强语义连接
                else:
                    color = "#8E44AD"  # 紫色 - 弱语义连接
            else:
                if weight >= 1.5:
                    color = "#3498DB"  # 蓝色 - 强嵌入连接
                else:
                    color = "#95A5A6"  # 灰色 - 弱嵌入连接
            
            edge_description = self.create_edge_description(source, target, edge_data)
            
            net.add_edge(source, target,
                         color=color,
                         width=width,
                         title=edge_description)
        
        # 生成HTML
        html_content = net.generate_html()
        
        # 统计信息
        weight_stats = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
        semantic_edges = sum(1 for u, v in G.edges() if G[u][v].get('semantic_relations'))
        
        # 诊断信息
        diagnostics_html = ""
        if diagnostics:
            matched_entities, missing_entities, missing_in_graph = diagnostics
            data_source = "ChromaDB" if (ChromaStore and isinstance(self.chunk_store, ChromaStore)) else "传统Embedding Store"
            diagnostics_html = f"""
            <h4>🔍 数据源诊断</h4>
            <p><strong>数据源类型:</strong> {data_source}</p>
            <p><strong>✅ 成功匹配:</strong> {len(matched_entities)}</p>
            <p><strong>❌ 缺失实体:</strong> {len(missing_entities)}</p>
            <p><strong>📊 数据统计:</strong></p>
            <ul>
                <li>文档块: {self.chunk_store.count()}</li>
                <li>实体: {self.entity_store.count()}</li>
                <li>事实: {self.fact_store.count()}</li>
            </ul>
            """
        
        # 添加统计信息面板
        data_source = "ChromaDB" if (ChromaStore and isinstance(self.chunk_store, ChromaStore)) else "传统Embedding Store"
        data_source_icon = "🗄️" if (ChromaStore and isinstance(self.chunk_store, ChromaStore)) else "📁"
        stats_panel = f"""
        <div style="position: fixed; top: 10px; right: 10px; width: 380px; background: white; 
                    border: 1px solid #ccc; padding: 15px; border-radius: 8px; font-family: Arial; z-index: 1000; max-height: 80vh; overflow-y: auto;">
            <h3>🎭 HippoRAG知识图谱</h3>
            <p><strong>数据源:</strong> {data_source_icon} {data_source}</p>
            <p><strong>工作目录:</strong> {os.path.basename(self.working_dir)}</p>
            <p><strong>节点数量:</strong> {G.number_of_nodes()}</p>
            <p><strong>边数量:</strong> {G.number_of_edges()}</p>
            <p><strong>实体节点:</strong> {len(entity_nodes)}</p>
            <p><strong>文档块节点:</strong> {len(chunk_nodes)}</p>
            <p><strong>语义边:</strong> {semantic_edges}</p>
            <p><strong>嵌入边:</strong> {G.number_of_edges() - semantic_edges}</p>
            {f"<p><strong>权重范围:</strong> {min(weight_stats):.3f} - {max(weight_stats):.3f}</p>" if weight_stats else ""}
            
            {diagnostics_html}
            
            <h4>🔥 度最高的节点</h4>
            <ul>
            {''.join([f"<li>{node[-8:]}: {deg}</li>" for node, deg in sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]])}
            </ul>
            
            <h4>🎨 节点颜色说明</h4>
            <ul>
                <li>🔴 红色: 人物实体/缺失实体</li>
                <li>🟢 青色: 地点实体</li>
                <li>🔵 蓝色: 物品实体</li>
                <li>🟢 绿色: 职业实体</li>
                <li>🟣 紫色: 事件实体</li>
                <li>🟡 黄色: 其他实体</li>
                <li>⚫ 黑色: 文档块</li>
            </ul>
            
            <h4>🔗 边颜色说明</h4>
            <ul>
                <li>🔴 红色: 强语义连接</li>
                <li>🟣 紫色: 弱语义连接</li>
                <li>🔵 蓝色: 强嵌入连接</li>
                <li>⚫ 灰色: 弱嵌入连接</li>
            </ul>
            
            <h4>💡 数据源特色</h4>
            <ul>
                {'<li>🗄️ 基于ChromaDB向量存储</li><li>🚀 高效的向量检索</li><li>📊 持久化存储</li><li>🔍 语义搜索能力</li><li>⚡ 避免内存爆炸</li>' if (ChromaStore and isinstance(self.chunk_store, ChromaStore)) else '<li>📁 基于传统Embedding Store</li><li>🔄 自动适配兼容模式</li><li>📋 Parquet格式存储</li><li>⚡ 快速文件加载</li><li>🔧 向后兼容性</li>'}
            </ul>
        </div>
        """
        
        # 插入统计面板
        html_content = html_content.replace('<body>', f'<body>{stats_panel}')
        
        # 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ ChromaDB图谱可视化已保存到: {output_file}")
        
        return G

def main():
    """主函数"""
    # 直接使用 test_outputs/deepseek-v3_bge-m3 目录
    working_dir = "./outputs/tenant_1/deepseek-v3_bge-m3"
    
    if not os.path.exists(working_dir):
        print(f"❌ 工作目录不存在: {working_dir}")
        return
    
    # 检查是否包含ChromaDB数据
    chroma_chunk_path = os.path.join(working_dir, "chroma_chunk_embeddings")
    traditional_chunk_path = os.path.join(working_dir, "chunk_embeddings")
    
    if not (os.path.exists(chroma_chunk_path) or os.path.exists(traditional_chunk_path)):
        print(f"❌ 目录中未找到ChromaDB或传统embedding store数据: {working_dir}")
        print("请确保目录包含以下数据之一:")
        print(f"  - ChromaDB数据: {chroma_chunk_path}")
        print(f"  - 传统embedding store数据: {traditional_chunk_path}")
        return
    
    print(f"📁 使用工作目录: {working_dir}")
    
    # 创建可视化器
    visualizer = ChromaGraphVisualizer(working_dir)
    
    # 加载数据
    if not visualizer.load_chroma_stores():
        return
    
    if not visualizer.load_graph():
        return
    
    # 加载OpenIE数据（可选）
    visualizer.load_openie_data()
    
    # 构建NetworkX图
    result = visualizer.build_networkx_graph()
    if result is None:
        return
    
    G, diagnostics = result
    
    # 生成可视化
    data_source_prefix = "chroma" if (ChromaStore and isinstance(visualizer.chunk_store, ChromaStore)) else "hipporag"
    output_file = f"{data_source_prefix}_knowledge_graph_{os.path.basename(working_dir)}.html"
    visualizer.visualize_graph(G, output_file, diagnostics)
    
    data_source_type = "ChromaDB" if (ChromaStore and isinstance(visualizer.chunk_store, ChromaStore)) else "传统Embedding Store"
    print(f"\n🎉 完成！请打开 {output_file} 查看HippoRAG知识图谱")
    print(f"✨ {data_source_type}图谱特色:")
    
    if ChromaStore and isinstance(visualizer.chunk_store, ChromaStore):
        print("   - 🗄️ 基于ChromaDB向量存储")
        print("   - 🚀 高效的向量检索和相似度计算")
        print("   - 📊 持久化存储，避免内存爆炸")
        print("   - 🔍 支持语义搜索和相似度查询")
        print("   - ⚡ 可扩展到大规模知识图谱")
    else:
        print("   - 📁 基于传统Embedding Store")
        print("   - 🔄 自动适配兼容模式")
        print("   - 📋 Parquet格式存储")
        print("   - ⚡ 快速文件加载")
        print("   - 🔧 向后兼容性保证")
    
    print("   - 🎯 保持与原始图谱结构的完全兼容")

if __name__ == "__main__":
    main() 