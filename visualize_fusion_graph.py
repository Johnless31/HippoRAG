#!/usr/bin/env python3
"""
融合可视化HippoRAG图谱 - 结合graph.pickle和OpenIE数据
展示最真实、最丰富的知识图谱信息
"""

import pickle
import json
import os
from pyvis.network import Network
import networkx as nx
from collections import defaultdict, Counter

def load_hipporag_graph(pickle_file):
    """加载HippoRAG的igraph图谱"""
    try:
        import igraph as ig
    except ImportError:
        print("❌ 需要安装igraph: pip install igraph")
        return None
    
    if not os.path.exists(pickle_file):
        print(f"❌ 找不到图谱文件: {pickle_file}")
        return None
    
    with open(pickle_file, 'rb') as f:
        igraph_obj = pickle.load(f)
    
    print(f"✅ 成功加载HippoRAG图谱: {igraph_obj.summary()}")
    return igraph_obj

def load_openie_data(json_file):
    """加载OpenIE结果数据"""
    if not os.path.exists(json_file):
        print(f"❌ 找不到OpenIE文件: {json_file}")
        return None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        openie_data = json.load(f)
    
    print(f"✅ 成功加载OpenIE数据: {len(openie_data['docs'])} 个文档")
    return openie_data

def extract_openie_semantics(openie_data):
    """从OpenIE数据中提取语义信息"""
    entity_contexts = defaultdict(list)
    entity_relations = defaultdict(list)
    entity_types = defaultdict(set)
    relation_mappings = defaultdict(list)
    
    # 统计实体频率（包括三元组中的概念）
    entity_frequencies = Counter()
    
    for doc in openie_data['docs']:
        passage = doc['passage']
        entities = doc['extracted_entities']
        triples = doc['extracted_triples']
        
        # 收集所有实体
        all_entities = set(entities)
        for triple in triples:
            subject, relation, obj = triple
            all_entities.add(subject)
            all_entities.add(obj)
        
        # 记录上下文
        for entity in all_entities:
            entity_contexts[entity].append(passage)
            entity_frequencies[entity] += 1
        
        # 记录关系
        for triple in triples:
            subject, relation, obj = triple
            relation_str = f"{subject} → {relation} → {obj}"
            entity_relations[subject].append(relation_str)
            entity_relations[obj].append(relation_str)
            relation_mappings[(subject, obj)].append((relation, passage))
            relation_mappings[(obj, subject)].append((relation, passage))  # 双向记录
    
    # 推断实体类型
    for entity in entity_frequencies:
        relations = entity_relations[entity]
        inferred_types = infer_entity_type(entity, relations)
        entity_types[entity].update(inferred_types)
    
    return entity_contexts, entity_relations, entity_types, entity_frequencies, relation_mappings

def infer_entity_type(entity_name, relations):
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
        elif "is part of" in relation_str:
            if entity_name in relation_str.split(" → ")[0]:
                entity_types.add("Place/Region")
            else:
                entity_types.add("Larger Region")
    
    # 根据实体名称推断
    if "County" in entity_name:
        entity_types.add("Administrative Region")
    elif entity_name in ["Cinderella", "prince"]:
        entity_types.add("Character")
    elif entity_name in ["kingdom", "royal ball"]:
        entity_types.add("Place/Event")
    elif "slipper" in entity_name:
        entity_types.add("Object")
    elif entity_name == "politician":
        entity_types.add("Profession/Role")
    
    return list(entity_types)

def igraph_to_networkx_with_semantics(igraph_obj, openie_semantics):
    """将igraph转换为NetworkX并融合语义信息"""
    entity_contexts, entity_relations, entity_types, entity_frequencies, relation_mappings = openie_semantics
    
    # 创建NetworkX图
    G = nx.Graph()
    
    # 诊断信息
    missing_entities = []
    matched_entities = []
    
    # 添加节点并融合语义信息
    for i in range(igraph_obj.vcount()):
        node_attrs = {}
        for attr in igraph_obj.vertex_attributes():
            node_attrs[attr] = igraph_obj.vs[i][attr]
        
        node_id = node_attrs.get('name', f'node_{i}')
        node_content = node_attrs.get('content', '').lower()
        
        # 融合OpenIE语义信息
        if node_id.startswith('entity-'):
            # 查找对应的实体名称
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
                # 检查是否是中文实体被错误处理
                is_likely_chinese = node_content == '' and node_id.startswith('entity-')
                if is_likely_chinese:
                    missing_entities.append((node_id, node_content, "可能是中文实体被错误处理"))
                else:
                    missing_entities.append((node_id, node_content, "未知原因"))
                
                node_attrs['semantic_name'] = node_content if node_content else "未知实体"
                node_attrs['contexts'] = []
                node_attrs['relations'] = []
                node_attrs['inferred_types'] = []
                node_attrs['frequency'] = 0
        
        G.add_node(node_id, **node_attrs)
    
    # 添加边并融合语义信息
    for i in range(igraph_obj.ecount()):
        edge = igraph_obj.es[i]
        source_idx = edge.source
        target_idx = edge.target
        
        source_name = igraph_obj.vs[source_idx]['name']
        target_name = igraph_obj.vs[target_idx]['name']
        
        edge_attrs = {}
        for attr in igraph_obj.edge_attributes():
            edge_attrs[attr] = edge[attr]
        
        # 融合语义关系信息
        source_content = igraph_obj.vs[source_idx]['content']
        target_content = igraph_obj.vs[target_idx]['content']
        
        # 查找语义关系
        semantic_relations = []
        if (source_content, target_content) in relation_mappings:
            semantic_relations.extend(relation_mappings[(source_content, target_content)])
        if (target_content, source_content) in relation_mappings:
            semantic_relations.extend(relation_mappings[(target_content, source_content)])
        
        edge_attrs['semantic_relations'] = semantic_relations
        
        G.add_edge(source_name, target_name, **edge_attrs)
    
    # 打印诊断信息
    print(f"\n🔍 实体匹配诊断:")
    print(f"   ✅ 成功匹配的实体: {len(matched_entities)}")
    for node_id, semantic_name, graph_content in matched_entities:
        print(f"     - {node_id[:20]}... -> '{semantic_name}' (图谱: '{graph_content}')")
    
    if missing_entities:
        print(f"   ❌ 缺失的实体: {len(missing_entities)}")
        for node_id, graph_content, reason in missing_entities:
            print(f"     - {node_id[:20]}... -> '{graph_content}' ({reason})")
    
    # 检查OpenIE中有但图谱中没有的实体
    graph_entities = set(attr.get('content', '').lower() for _, attr in G.nodes(data=True) if attr.get('name', '').startswith('entity-'))
    openie_entities = set(entity.lower() for entity in entity_frequencies.keys())
    missing_in_graph = openie_entities - graph_entities
    
    if missing_in_graph:
        print(f"   ⚠️  OpenIE中存在但图谱中缺失的实体: {len(missing_in_graph)}")
        # 找到原始大小写形式的实体名称
        entity_lowercase_to_original = {entity.lower(): entity for entity in entity_frequencies.keys()}
        for entity_lower in missing_in_graph:
            original_entity = entity_lowercase_to_original.get(entity_lower, entity_lower)
            print(f"     - '{original_entity}' (可能由于text_processing函数问题)")
    
    # 返回图和诊断信息
    diagnostics = (matched_entities, missing_entities, missing_in_graph)
    return G, diagnostics

def create_fusion_description(node_id, node_data, degrees):
    """创建融合的节点描述"""
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
        desc_parts.append(f"📋 图谱ID: {node_id}")
        desc_parts.append(f"🔑 Hash: {node_data.get('hash_id', 'N/A')[:12]}...")
        
        if contexts:
            desc_parts.append(f"📖 上下文:")
            for i, context in enumerate(set(contexts[:3]), 1):  # 最多显示3个
                desc_parts.append(f"  {i}. {context}")
        
        if relations:
            desc_parts.append(f"🔗 语义关系:")
            unique_relations = list(set(relations[:5]))  # 最多显示5个
            for i, relation in enumerate(unique_relations, 1):
                desc_parts.append(f"  {i}. {relation}")
    
    elif node_id.startswith('chunk-'):
        content = node_data.get('content', 'Unknown')
        desc_parts.append(f"📄 文档块: {content}")
        desc_parts.append(f"🔗 图谱度: {degrees[node_id]}")
        desc_parts.append(f"📋 图谱ID: {node_id}")
        desc_parts.append(f"🔑 Hash: {node_data.get('hash_id', 'N/A')[:12]}...")
    
    return "\n".join(desc_parts)

def create_fusion_edge_description(source, target, edge_data):
    """创建融合的边描述"""
    weight = edge_data.get('weight', 1.0)
    semantic_relations = edge_data.get('semantic_relations', [])
    
    desc_parts = []
    desc_parts.append(f"🔗 连接: {source} ↔ {target}")
    desc_parts.append(f"⚖️ 嵌入权重: {weight:.3f}")
    
    if semantic_relations:
        desc_parts.append(f"📝 语义关系:")
        for i, (relation, context) in enumerate(semantic_relations[:3], 1):  # 最多显示3个
            desc_parts.append(f"  {i}. {relation}")
            desc_parts.append(f"     来源: {context[:50]}...")
    else:
        desc_parts.append(f"📝 语义关系: 基于嵌入相似度连接")
    
    return "\n".join(desc_parts)

def get_entity_color(entity_content, inferred_types):
    """根据实体内容和类型确定颜色"""
    content_lower = entity_content.lower()
    
    if 'politician' in content_lower or 'Profession/Role' in inferred_types:
        return "#96CEB4"  # 绿色 - 职业
    elif any(name in content_lower for name in ['cinderella', 'prince', 'oliver badman', 'george rankin', 'thomas marwick', 'erik hort', 'marina']):
        return "#FF6B6B"  # 红色 - 人物
    elif any(place in content_lower for place in ['montebello', 'minsk', 'rockland county', 'kingdom']) or any(t in inferred_types for t in ['Place', 'Administrative Region']):
        return "#4ECDC4"  # 青色 - 地点
    elif 'slipper' in content_lower or 'Object' in inferred_types:
        return "#45B7D1"  # 蓝色 - 物品
    elif 'royal ball' in content_lower or any(t in inferred_types for t in ['Place/Event', 'Event']):
        return "#DDA0DD"  # 紫色 - 事件
    else:
        return "#FFEAA7"  # 黄色 - 其他

def visualize_fusion_graph(G, output_file, diagnostics=None):
    """可视化融合图谱"""
    
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
        # 计算节点大小
        node_degree = degrees[node_id]
        frequency = node_data.get('frequency', 0)
        base_size = 18
        size = base_size + node_degree * 3 + frequency * 2
        
        # 设置节点颜色和标签
        if node_id.startswith('entity-'):
            content = node_data.get('semantic_name', node_data.get('content', ''))
            inferred_types = node_data.get('inferred_types', [])
            
            # 特殊处理缺失实体
            if content == "未知实体" or content == "":
                color = "#FF0000"  # 红色 - 缺失实体
                label = "❌未知"
            else:
                color = get_entity_color(content, inferred_types)
                label = content
        elif node_id.startswith('chunk-'):
            color = "#000000"  # 黑色 - 文档
            label = f"Doc-{node_id[-8:]}"
        else:
            color = "#B2B2B2"  # 灰色 - 未知
            label = node_id
        
        # 创建融合描述
        description = create_fusion_description(node_id, node_data, degrees)
        
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
        
        # 边宽度和颜色
        width = max(2, int(weight * 4))
        
        # 根据权重和语义关系确定颜色
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
        
        # 创建边描述
        edge_description = create_fusion_edge_description(source, target, edge_data)
        
        net.add_edge(source, target,
                     color=color,
                     width=width,
                     title=edge_description)
    
    # 生成HTML
    html_content = net.generate_html()
    
    # 计算统计信息
    weight_stats = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    semantic_edges = sum(1 for u, v in G.edges() if G[u][v].get('semantic_relations'))
    
    # 诊断信息
    diagnostics_html = ""
    if diagnostics:
        matched_entities, missing_entities, missing_in_graph = diagnostics
        diagnostics_html = f"""
        <h4>🔍 实体匹配诊断</h4>
        <p><strong>✅ 成功匹配:</strong> {len(matched_entities)}</p>
        <p><strong>❌ 缺失实体:</strong> {len(missing_entities)}</p>
        <p><strong>⚠️ OpenIE缺失:</strong> {len(missing_in_graph)}</p>
        
        {f'''<div style="margin-top: 10px; padding: 10px; background: #ffebee; border-radius: 5px;">
        <strong>⚠️ 检测到问题:</strong><br>
        OpenIE中有{len(missing_in_graph)}个实体在图谱中缺失，这可能是由于text_processing函数的问题导致的。<br>
        <strong>缺失实体:</strong> {", ".join(list(missing_in_graph)[:5])}{"..." if len(missing_in_graph) > 5 else ""}
        </div>''' if missing_in_graph else ""}
        """
    
    # 添加统计信息面板
    stats_panel = f"""
    <div style="position: fixed; top: 10px; right: 10px; width: 380px; background: white; 
                border: 1px solid #ccc; padding: 15px; border-radius: 8px; font-family: Arial; z-index: 1000; max-height: 80vh; overflow-y: auto;">
        <h3>🎭 融合知识图谱</h3>
        <p><strong>节点数量:</strong> {G.number_of_nodes()}</p>
        <p><strong>边数量:</strong> {G.number_of_edges()}</p>
        <p><strong>实体节点:</strong> {len(entity_nodes)}</p>
        <p><strong>文档块节点:</strong> {len(chunk_nodes)}</p>
        <p><strong>语义边:</strong> {semantic_edges}</p>
        <p><strong>嵌入边:</strong> {G.number_of_edges() - semantic_edges}</p>
        <p><strong>权重范围:</strong> {min(weight_stats):.3f} - {max(weight_stats):.3f}</p>
        
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
        
        <h4>💡 特色功能</h4>
        <ul>
            <li>✨ 融合图谱结构和语义</li>
            <li>🎯 显示实体类型和上下文</li>
            <li>📊 展示嵌入权重和语义关系</li>
            <li>🔍 悬停查看详细信息</li>
            <li>⚠️ 自动检测实体匹配问题</li>
        </ul>
    </div>
    """
    
    # 插入统计面板
    html_content = html_content.replace('<body>', f'<body>{stats_panel}')
    
    # 保存文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 融合图谱可视化已保存到: {output_file}")
    
    return G

def main():
    """主函数"""
    # 文件路径
    pickle_file = "outputs/deepseek-v3_bge-m3/graph.pickle"
    openie_file = "outputs/openie_results_ner_deepseek-v3.json"
    
    print("📚 加载HippoRAG图谱和OpenIE数据...")
    
    # 加载数据
    igraph_obj = load_hipporag_graph(pickle_file)
    openie_data = load_openie_data(openie_file)
    
    if igraph_obj is None or openie_data is None:
        return
    
    print("🔍 提取OpenIE语义信息...")
    openie_semantics = extract_openie_semantics(openie_data)
    
    print("🔄 转换为融合NetworkX图...")
    G, diagnostics = igraph_to_networkx_with_semantics(igraph_obj, openie_semantics)
    
    print("🎨 生成融合可视化...")
    output_file = "fusion_knowledge_graph.html"
    visualize_fusion_graph(G, output_file, diagnostics)
    
    print(f"\n🎉 完成！请打开 {output_file} 查看融合知识图谱")
    print("✨ 融合图谱特色:")
    print("   - 🧠 保持HippoRAG的真实图谱结构")
    print("   - 📝 融合OpenIE的语义关系描述")
    print("   - 🎯 展示实体类型和上下文信息")
    print("   - ⚖️ 显示嵌入权重和语义关系")
    print("   - 🔍 提供最丰富的交互式详情")

if __name__ == "__main__":
    main() 