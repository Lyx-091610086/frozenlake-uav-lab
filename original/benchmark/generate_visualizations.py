#!/usr/bin/env python3
"""
可视化生成入口脚本
调用visualization包中的可视化生成器
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入并运行可视化生成器
if __name__ == '__main__':
    from visualization.generate_visualizations import main
    sys.exit(main())
