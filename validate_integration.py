"""
Validation script to test that all federated components are working correctly.
Run this to verify the integration before running full training.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
import sys

def test_imports():
    """Test that all modules can be imported"""
    print("✓ Testing imports...")
    try:
        from federated.client import FederatedLFFClient
        from federated.server import FederatedLFFServer
        from federated.utils import (aggregate_models, ClientManager, 
                                     get_client_data_distribution)
        from local_reweighting import (LocalReweightingManager, 
                                       ConflictDetectionReweighting,
                                       AdaptiveLocalReweighting)
        from module.loss import GeneralizedCELoss
        from module.util import get_backbone
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("\n✓ Testing model creation...")
    try:
        from module.util import get_backbone
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_classes = 10
        
        # Create models
        model_b = get_backbone('MLP', num_classes)
        model_d = get_backbone('MLP', num_classes)
        
        print(f"  ✓ Model B created: {sum(p.numel() for p in model_b.parameters())} params")
        print(f"  ✓ Model D created: {sum(p.numel() for p in model_d.parameters())} params")
        return True, model_b, model_d
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False, None, None


def test_loss_functions():
    """Test loss function creation"""
    print("\n✓ Testing loss functions...")
    try:
        from module.loss import GeneralizedCELoss
        import torch.nn as nn
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        bias_criterion = GeneralizedCELoss(q=0.7)
        
        # Test with dummy batch
        batch_size = 32
        num_classes = 10
        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        loss1 = criterion(logits, labels)
        loss2 = bias_criterion(logits, labels)
        
        print(f"  ✓ CE loss: shape={loss1.shape}, mean={loss1.mean():.4f}")
        print(f"  ✓ GCE loss: shape={loss2.shape}, mean={loss2.mean():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Loss function test failed: {e}")
        return False


def test_reweighting_methods():
    """Test all reweighting methods"""
    print("\n✓ Testing local reweighting methods...")
    try:
        from local_reweighting import (LocalReweightingManager,
                                       ConflictDetectionReweighting,
                                       AdaptiveLocalReweighting)
        
        num_samples = 100
        batch_size = 32
        num_classes = 10
        
        # Create dummy data
        indices = torch.arange(batch_size)
        logits_b = torch.randn(batch_size, num_classes)
        logits_d = torch.randn(batch_size, num_classes)
        losses = torch.rand(batch_size)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        # Test 1: Loss-based reweighting
        print("  Testing loss-based reweighting...")
        loss_mgr = LocalReweightingManager('loss_based')
        loss_mgr.initialize(num_samples)
        weights = loss_mgr.update_weights(indices, losses, labels)
        print(f"    ✓ Weights: shape={weights.shape}, sum={weights.sum():.4f}")
        
        # Test 2: Conflict detection
        print("  Testing conflict detection reweighting...")
        conflict_mgr = ConflictDetectionReweighting()
        conflict_mgr.initialize(num_samples)
        weights, conflict_mask = conflict_mgr.detect_and_reweight(
            indices, logits_b, logits_d, labels
        )
        n_conflict = conflict_mask.sum().item()
        print(f"    ✓ Weights: shape={weights.shape}")
        print(f"    ✓ Conflicts detected: {n_conflict}/{batch_size}")
        
        # Test 3: Adaptive reweighting
        print("  Testing adaptive reweighting...")
        adaptive_mgr = AdaptiveLocalReweighting()
        adaptive_mgr.initialize(num_samples)
        weight_dict = adaptive_mgr.compute_adaptive_weights(
            indices, logits_b, logits_d, losses, losses, labels,
            progress_ratio=0.5
        )
        print(f"    ✓ Adaptive weights: {weight_dict['weights'].shape}")
        print(f"    ✓ Alpha conflict: {weight_dict['alpha_conflict']:.4f}")
        print(f"    ✓ Alpha loss: {weight_dict['alpha_loss']:.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Reweighting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_client_server():
    """Test client-server communication"""
    print("\n✓ Testing client-server framework...")
    try:
        from federated.client import FederatedLFFClient
        from federated.server import FederatedLFFServer
        from module.util import get_backbone
        from module.loss import GeneralizedCELoss
        import torch.nn as nn
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_classes = 10
        
        # Create models
        model_b = get_backbone('MLP', num_classes)
        model_d = get_backbone('MLP', num_classes)
        criterion = nn.CrossEntropyLoss(reduction='none')
        bias_criterion = GeneralizedCELoss(q=0.7)
        
        # Create dummy dataset
        num_samples = 256
        X = torch.randn(num_samples, 3, 28, 28)
        y = torch.randint(0, num_classes, (num_samples,))
        dataset = TensorDataset(X, y)
        
        # Create server
        print("  Creating server...")
        global_b = get_backbone('MLP', num_classes)
        global_d = get_backbone('MLP', num_classes)
        server = FederatedLFFServer(global_b, global_d, 2, device=device)
        print("  ✓ Server created")
        
        # Create clients
        print("  Creating clients...")
        client1_dataset = Subset(dataset, list(range(128)))
        client2_dataset = Subset(dataset, list(range(128, 256)))
        
        client1 = FederatedLFFClient(
            0, get_backbone('MLP', num_classes), get_backbone('MLP', num_classes),
            client1_dataset, criterion, bias_criterion, device=device, num_classes=num_classes
        )
        client2 = FederatedLFFClient(
            1, get_backbone('MLP', num_classes), get_backbone('MLP', num_classes),
            client2_dataset, criterion, bias_criterion, device=device, num_classes=num_classes
        )
        print("  ✓ Clients created")
        
        # Test communication
        print("  Testing model sending...")
        package = server.send_model_to_client(0)
        print(f"    ✓ Package keys: {list(package.keys())}")
        
        print("  Testing model loading...")
        client1.load_model_state(package['model_b_state'], package['model_d_state'])
        print("    ✓ Models loaded successfully")
        
        print("  Testing model state retrieval...")
        state_b, state_d = client1.get_model_state()
        print(f"    ✓ State B keys: {len(state_b)} parameters")
        print(f"    ✓ State D keys: {len(state_d)} parameters")
        
        return True
    except Exception as e:
        print(f"  ✗ Client-server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregation():
    """Test model aggregation"""
    print("\n✓ Testing model aggregation...")
    try:
        from federated.server import FederatedLFFServer
        from federated.utils import aggregate_models
        from module.util import get_backbone
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_classes = 10
        
        # Create some model states
        model = get_backbone('MLP', num_classes)
        state1 = {k: v.clone() for k, v in model.state_dict().items()}
        state2 = {k: v.clone() + 0.1 for k, v in model.state_dict().items()}
        state3 = {k: v.clone() + 0.2 for k, v in model.state_dict().items()}
        
        # Test aggregation
        aggregated = aggregate_models([state1, state2, state3])
        print(f"  ✓ Aggregated {len([state1, state2, state3])} models")
        print(f"  ✓ Result has {len(aggregated)} parameters")
        
        # Verify it's between the originals
        for key in state1.keys():
            min_val = min(state1[key].min(), state2[key].min(), state3[key].min())
            max_val = max(state1[key].max(), state2[key].max(), state3[key].max())
            agg_val = aggregated[key]
            in_range = (agg_val >= min_val - 0.01).all() and (agg_val <= max_val + 0.01).all()
            if not in_range:
                print(f"  ✗ Aggregation out of range for {key}")
                return False
        
        print("  ✓ Aggregation produces sensible results")
        return True
    except Exception as e:
        print(f"  ✗ Aggregation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_distribution():
    """Test client data distribution"""
    print("\n✓ Testing data distribution...")
    try:
        from federated.utils import get_client_data_distribution
        import torch
        
        # Create dummy dataset
        num_samples = 1000
        X = torch.randn(num_samples)
        indices = list(range(num_samples))
        
        class DummyDataset:
            def __len__(self):
                return num_samples
            def __getitem__(self, idx):
                return torch.tensor([idx % 10])  # 10 classes
        
        dataset = DummyDataset()
        
        # Test IID distribution
        print("  Testing IID distribution...")
        client_indices_iid = get_client_data_distribution(dataset, 5, 'iid')
        print(f"    ✓ 5 clients, IID distribution:")
        for i, indices in enumerate(client_indices_iid):
            print(f"      Client {i}: {len(indices)} samples")
        
        # Test non-IID distribution
        print("  Testing non-IID distribution...")
        client_indices_noniid = get_client_data_distribution(dataset, 5, 'non-iid')
        print(f"    ✓ 5 clients, non-IID distribution:")
        for i, indices in enumerate(client_indices_noniid):
            print(f"      Client {i}: {len(indices)} samples")
        
        return True
    except Exception as e:
        print(f"  ✗ Data distribution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end():
    """Test a simple end-to-end federated round"""
    print("\n✓ Testing end-to-end federated round...")
    try:
        from federated.client import FederatedLFFClient
        from federated.server import FederatedLFFServer
        from module.util import get_backbone
        from module.loss import GeneralizedCELoss
        import torch.nn as nn
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        num_classes = 10
        
        # Setup
        print("  Setting up server and clients...")
        global_b = get_backbone('MLP', num_classes)
        global_d = get_backbone('MLP', num_classes)
        server = FederatedLFFServer(global_b, global_d, 2, device=device)
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        bias_criterion = GeneralizedCELoss(q=0.7)
        
        # Create dummy datasets
        X1 = torch.randn(128, 3, 28, 28)
        y1 = torch.randint(0, num_classes, (128,))
        X2 = torch.randn(128, 3, 28, 28)
        y2 = torch.randint(0, num_classes, (128,))
        
        from data.util import IdxDataset
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return self.X[idx], torch.tensor([self.y[idx], 0]), ""
        
        ds1 = IdxDataset(DummyDataset(X1, y1))
        ds2 = IdxDataset(DummyDataset(X2, y2))
        
        # Create clients
        client1 = FederatedLFFClient(
            0, get_backbone('MLP', num_classes), get_backbone('MLP', num_classes),
            ds1, criterion, bias_criterion, device=device, num_classes=num_classes
        )
        client2 = FederatedLFFClient(
            1, get_backbone('MLP', num_classes), get_backbone('MLP', num_classes),
            ds2, criterion, bias_criterion, device=device, num_classes=num_classes
        )
        
        server.register_client(0, len(ds1))
        server.register_client(1, len(ds2))
        
        clients = [client1, client2]
        
        # Simulate one federated round
        print("  Running federated round...")
        
        # Send models to clients
        for idx, client in enumerate(clients):
            package = server.send_model_to_client(idx)
            client.load_model_state(package['model_b_state'], package['model_d_state'])
        print("    ✓ Models sent to clients")
        
        # Local training
        client_updates = []
        for idx, client in enumerate(clients):
            metrics = client.local_train(num_epochs=1, batch_size=32)
            state_b, state_d = client.get_model_state()
            update = {
                'client_id': idx,
                'model_b_state': state_b,
                'model_d_state': state_d,
                'data_size': len(client.local_dataset),
                'metrics': metrics
            }
            client_updates.append(update)
        print("    ✓ Local training completed")
        
        # Aggregation
        server.aggregate_client_models(client_updates)
        print("    ✓ Model aggregation completed")
        
        # Get stats
        stats = server.get_statistics()
        print(f"    ✓ Server stats: {stats['global_updates']} global updates")
        
        print("  ✓ End-to-end federated round successful!")
        return True
    except Exception as e:
        print(f"  ✗ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all validation tests"""
    print("="*60)
    print("FEDERATED LFF - INTEGRATION VALIDATION")
    print("="*60)
    
    results = []
    
    # Basic tests
    results.append(("Imports", test_imports()))
    
    success, model_b, model_d = test_model_creation()
    results.append(("Model Creation", success))
    
    results.append(("Loss Functions", test_loss_functions()))
    results.append(("Reweighting Methods", test_reweighting_methods()))
    results.append(("Client-Server", test_client_server()))
    results.append(("Aggregation", test_aggregation()))
    results.append(("Data Distribution", test_data_distribution()))
    results.append(("End-to-End", test_end_to_end()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Integration is working correctly.")
        print("\nYou can now run:")
        print("  python lff_federated.py --num_clients 3 --num_rounds 5")
        return 0
    else:
        print("✗ Some tests failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
