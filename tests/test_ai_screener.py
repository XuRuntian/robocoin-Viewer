import pytest
import numpy as np
from unittest.mock import patch
from src.core.ai_screener import AIScreener

@pytest.fixture
def mock_screener():
    # 阻断 __init__，防止在跑自动化测试时下载 CLIP 模型
    with patch.object(AIScreener, '__init__', return_value=None):
        screener = AIScreener()
        yield screener

def test_detect_outliers_logic(mock_screener):
    """测试离群检测的核心数学逻辑是否正确"""
    # 1. 构造假路径
    normal_paths = [f"data/normal_{i}.hdf5" for i in range(10)]
    outlier_path = "data/BAD_DATA.hdf5"
    all_paths = normal_paths + [outlier_path]

    # 2. 构造假特征 (Mock Embeddings)
    # 正常数据：特征向量彼此非常接近（全 1 附近）
    mock_embeddings = {
        path: np.ones(512) + np.random.normal(0, 0.05, 512) 
        for path in normal_paths
    }
    # 异常数据：特征向量方向完全不同（全 -1 附近）
    mock_embeddings[outlier_path] = -np.ones(512)

    # 3. 执行测试：Patch 掉 extract_embeddings 方法，直接返回假特征
    with patch.object(mock_screener, 'extract_embeddings', return_value=mock_embeddings):
        # 宽容的参数：抓取最底部的 10%，且相似度低于 0.5
        suspects = mock_screener.detect_outliers(
            all_paths, 
            outlier_ratio=0.1, 
            similarity_threshold=0.5
        )

        # 4. 断言 (Assertions)
        assert len(suspects) == 1, "应该只检测出一个异常数据"
        assert suspects[0] == outlier_path, "揪出的异常数据路径必须完全匹配"
