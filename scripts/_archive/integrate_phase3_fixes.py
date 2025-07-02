"""
Integrate Phase 3 Fixes
Integrate the working fixes for Phase 3 components
"""

import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_phase3_fixes():
    """
    Integrate the working Phase 3 fixes
    """
    logger.info("=== INTEGRATING PHASE 3 FIXES ===")
    
    results_dir = Path("results/business_cycle_analysis")
    
    # Files to replace with fixed versions (excluding timeline for now)
    fixes = {
        'risk_adjusted_heatmap_FIXED.html': 'risk_adjusted_heatmap.html',
        'factor_rotation_wheel_FIXED.html': 'factor_rotation_wheel.html',
        'momentum_persistence_analysis_FIXED.html': 'momentum_persistence_analysis.html'
    }
    
    integration_results = {}
    
    for fixed_file, original_file in fixes.items():
        fixed_path = results_dir / fixed_file
        original_path = results_dir / original_file
        
        if fixed_path.exists():
            # Replace original with fixed version
            shutil.copy2(fixed_path, original_path)
            integration_results[original_file] = True
            logger.info(f"✅ Integrated fix: {original_file}")
        else:
            integration_results[original_file] = False
            logger.warning(f"⚠️  Fixed file not found: {fixed_file}")
    
    # Summary
    successful_integrations = sum(integration_results.values())
    total_integrations = len(integration_results)
    
    logger.info(f"Integration Results: {successful_integrations}/{total_integrations} fixes applied")
    
    if successful_integrations == total_integrations:
        logger.info("🎉 ALL AVAILABLE FIXES SUCCESSFULLY INTEGRATED!")
        return True
    else:
        logger.warning("⚠️  Some fixes could not be integrated")
        return False

if __name__ == "__main__":
    integrate_phase3_fixes() 