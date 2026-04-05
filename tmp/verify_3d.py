from mol_prop_gnn.data.preprocessing import smiles_to_3d_graph
import torch

def test_3d_generation():
    smiles = "CCO" # Ethanol
    print(f"Testing 3D generation for: {smiles}")
    data = smiles_to_3d_graph(smiles)
    
    if data is None:
        print("FAILED: smiles_to_3d_graph returned None")
        return
        
    print(f"Success! Generated {data.num_nodes} atoms.")
    print(f"Pos shape: {data.pos.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Edge attr shape: {data.edge_attr.shape}")
    print(f"Atomic Numbers: {data.x.tolist()}")
    
    # Check if edge attributes are Euclidean distances
    row, col = data.edge_index
    expected_dist = (data.pos[row] - data.pos[col]).norm(dim=-1, keepdim=True)
    error = (data.edge_attr - expected_dist).abs().max().item()
    print(f"Distance verification error: {error:.2e}")

test_3d_generation()
