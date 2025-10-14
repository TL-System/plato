"""
Simple test script to verify server strategies can be imported and instantiated.

This tests that:
1. Strategy classes can be imported successfully
2. Strategies can be instantiated with parameters
3. ServerContext can be created
"""

print("Testing strategy imports and instantiation...\n")

# Test imports
try:
    from plato.servers.strategies import (
        ServerContext,
        FedAvgAggregationStrategy,
        FedNovaAggregationStrategy,
        FedAsyncAggregationStrategy,
        RandomSelectionStrategy,
        OortSelectionStrategy,
        AFLSelectionStrategy,
    )
    print("✓ All strategy imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test ServerContext instantiation
try:
    context = ServerContext()
    assert hasattr(context, 'server')
    assert hasattr(context, 'trainer')
    assert hasattr(context, 'algorithm')
    assert hasattr(context, 'current_round')
    assert hasattr(context, 'total_clients')
    assert hasattr(context, 'clients_per_round')
    assert hasattr(context, 'updates')
    assert hasattr(context, 'state')
    print("✓ ServerContext instantiated successfully")
except Exception as e:
    print(f"✗ ServerContext instantiation failed: {e}")
    exit(1)

# Test aggregation strategies
print("\nTesting aggregation strategies:")
try:
    agg1 = FedAvgAggregationStrategy()
    print(f"  ✓ FedAvgAggregationStrategy")

    agg2 = FedNovaAggregationStrategy()
    print(f"  ✓ FedNovaAggregationStrategy")

    agg3 = FedAsyncAggregationStrategy(mixing_hyperparameter=0.9)
    assert agg3.mixing_hyperparam == 0.9
    print(f"  ✓ FedAsyncAggregationStrategy(mixing_hyperparameter=0.9)")

    agg4 = FedAsyncAggregationStrategy(
        mixing_hyperparameter=0.8,
        adaptive_mixing=True,
        staleness_func_type='polynomial',
        staleness_func_params={'a': 0.5}
    )
    assert agg4.adaptive_mixing == True
    assert agg4.staleness_func_type == 'polynomial'
    print(f"  ✓ FedAsyncAggregationStrategy with adaptive mixing")

except Exception as e:
    print(f"  ✗ Aggregation strategy instantiation failed: {e}")
    exit(1)

# Test client selection strategies
print("\nTesting client selection strategies:")
try:
    sel1 = RandomSelectionStrategy()
    print(f"  ✓ RandomSelectionStrategy")

    sel2 = OortSelectionStrategy(exploration_factor=0.3)
    assert sel2.exploration_factor == 0.3
    print(f"  ✓ OortSelectionStrategy(exploration_factor=0.3)")

    sel3 = OortSelectionStrategy(
        exploration_factor=0.2,
        desired_duration=150.0,
        blacklist_num=15
    )
    assert sel3.desired_duration == 150.0
    assert sel3.blacklist_num == 15
    print(f"  ✓ OortSelectionStrategy with custom parameters")

    sel4 = AFLSelectionStrategy(alpha1=0.75, alpha2=0.01)
    assert sel4.alpha1 == 0.75
    assert sel4.alpha2 == 0.01
    print(f"  ✓ AFLSelectionStrategy(alpha1=0.75, alpha2=0.01)")

    sel5 = AFLSelectionStrategy(alpha1=0.8, alpha2=0.02, alpha3=0.15)
    assert sel5.alpha3 == 0.15
    print(f"  ✓ AFLSelectionStrategy with all alpha parameters")

except Exception as e:
    print(f"  ✗ Selection strategy instantiation failed: {e}")
    exit(1)

# Test strategy interfaces
print("\nTesting strategy interfaces:")
try:
    # All aggregation strategies should have aggregate_deltas method
    assert hasattr(agg1, 'aggregate_deltas')
    assert hasattr(agg2, 'aggregate_deltas')
    assert hasattr(agg3, 'aggregate_deltas')
    print("  ✓ All aggregation strategies have aggregate_deltas method")

    # FedAsync should also have aggregate_weights method
    assert hasattr(agg3, 'aggregate_weights')
    print("  ✓ FedAsync has aggregate_weights method")

    # All selection strategies should have select_clients method
    assert hasattr(sel1, 'select_clients')
    assert hasattr(sel2, 'select_clients')
    assert hasattr(sel4, 'select_clients')
    print("  ✓ All selection strategies have select_clients method")

    # All strategies should have setup method
    assert hasattr(agg1, 'setup')
    assert hasattr(sel1, 'setup')
    print("  ✓ All strategies have setup method")

except Exception as e:
    print(f"  ✗ Strategy interface check failed: {e}")
    exit(1)

print("\n" + "="*60)
print("STRATEGY INTEGRATION TEST - PASSED")
print("="*60)
print("✓ All strategy classes can be imported")
print("✓ All strategies can be instantiated with parameters")
print("✓ ServerContext works correctly")
print("✓ All required methods are present")
print("="*60)
