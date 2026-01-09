from typing import List, Dict

def build_topology(auxiliaries: List[Dict]) -> List[Dict]:
    """
    Build a topology tree based on the auxiliaries list.

    Returns:
        A list containing all root nodes (may form a forest).
        Each node structure:
        {
            "body_id": str,          # Body ID
            "children": List[Dict],  # List of child nodes
            "link_data": Dict        # Auxiliary data connecting to this node (None for roots)
        }
    """
    if not auxiliaries:
        return []

    # 1. Build Adjacency List and Children Set
    # Structure: parent_id -> [ (child_id, auxiliary_data), ... ]
    adj_list = {}
    all_bodies = set()
    children_bodies = set()

    for aux in auxiliaries:
        # Get Parent/Child IDs, convert to string for consistency
        # If parent_body is empty or N/A, usually treated as connected to World (Root)
        p_body = str(aux.get("parent_body")) if aux.get("parent_body") else "World"
        c_body = str(aux.get("current_body"))

        # Record all bodies involved
        all_bodies.add(p_body)
        all_bodies.add(c_body)

        # Record as a child node
        children_bodies.add(c_body)

        # Populate adjacency list
        if p_body not in adj_list:
            adj_list[p_body] = []

        adj_list[p_body].append({
            "child_id": c_body,
            "aux": aux
        })

    # 2. Find Root Nodes
    # Roots are bodies present in all_bodies but never appear as a child
    roots = list(all_bodies - children_bodies)

    # Safety check if there are no roots but bodies exist (e.g. circular dependency or empty roots)
    if not roots and all_bodies:
        # Heuristic: assume the first parent key is a root
        roots = [list(adj_list.keys())[0]]

    # 3. Recursive Tree Builder Helper
    def _build_node(body_id: str, link_data: Dict = None) -> Dict:
        node = {
            "body_id": body_id,
            "link_data": link_data,  # Auxiliary connecting to this body
            "children": []
        }

        # Find all children for this body
        if body_id in adj_list:
            for item in adj_list[body_id]:
                child_node = _build_node(item["child_id"], item["aux"])
                node["children"].append(child_node)

        return node

    # 4. Build Trees for all Roots
    forest = []
    for root_id in roots:
        forest.append(_build_node(root_id))

    return forest
