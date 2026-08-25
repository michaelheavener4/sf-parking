from scripts.benchmark_blockface_transaction_dynamics_v4 import normalized_mapping_sql


def test_blockface_mapping_normalizes_to_text():
    sql = normalized_mapping_sql()
    assert "post_id::text" in sql
    assert "blockface_id::text" in sql
