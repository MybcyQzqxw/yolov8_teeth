#!/usr/bin/env python3
"""
测试增强评估指标模块的导入和基本功能
"""

import sys
import os
sys.path.append('.')

def test_imports():
    """测试模块导入"""
    try:
        from utils.yolov8.metrics import calculate_f1_score, enhanced_metrics_analysis
        from utils.yolov8.per_class_evaluator import PerClassMetrics
        print('✅ 所有新增指标模块导入成功!')
        
        # 测试F1-Score计算
        f1 = calculate_f1_score(0.8, 0.7)
        print(f'✅ F1-Score计算测试: {f1:.3f}')
        
        print('✅ 增强评估指标系统已成功集成到项目中!')
        return True
    except ImportError as e:
        print(f'❌ 模块导入失败: {e}')
        return False
    except Exception as e:
        print(f'❌ 测试过程出错: {e}')
        return False

if __name__ == '__main__':
    success = test_imports()
    exit(0 if success else 1)
