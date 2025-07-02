"""
Comprehensive Phase 2 Verification for Phase 3 Readiness
Thorough validation of all Phase 2 outputs and data integrity before Phase 3 implementation
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensivePhase2Verifier:
    """
    Comprehensive verifier for Phase 2 completion and Phase 3 readiness
    """
    
    def __init__(self):
        self.data_dir = Path("data/processed")
        self.results_dir = Path("results/business_cycle_analysis")
        self.verification_results = {}
        self.critical_errors = []
        self.warnings = []
        
        logger.info("Comprehensive Phase 2 Verifier initialized")
    
    def verify_data_integrity(self):
        """
        Verify all required data files exist and have correct structure
        """
        logger.info("=== Verifying Data Integrity ===")
        
        required_files = {
            "phase2_outputs": [
                "phase2_regime_analysis.json",
                "phase2_performance_analysis.json", 
                "phase2_complete_summary.json"
            ],
            "aligned_datasets": [
                "aligned_master_dataset_FIXED.csv",
                "factor_returns_aligned_FIXED.csv",
                "regime_classifications_FIXED.csv"
            ]
        }
        
        file_verification = {}
        
        for category, files in required_files.items():
            file_verification[category] = {}
            
            for file in files:
                if category == "phase2_outputs" or category == "aligned_datasets":
                    file_path = self.results_dir / file
                else:
                    file_path = self.data_dir / file
                
                if file_path.exists():
                    try:
                        if file.endswith('.json'):
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            file_verification[category][file] = {
                                "exists": True,
                                "readable": True,
                                "size_kb": file_path.stat().st_size / 1024,
                                "keys": len(data) if isinstance(data, dict) else "N/A"
                            }
                        else:
                            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                            file_verification[category][file] = {
                                "exists": True,
                                "readable": True,
                                "size_kb": file_path.stat().st_size / 1024,
                                "shape": list(data.shape),
                                "date_range": f"{data.index.min()} to {data.index.max()}",
                                "columns": len(data.columns)
                            }
                    except Exception as e:
                        file_verification[category][file] = {
                            "exists": True,
                            "readable": False,
                            "error": str(e)
                        }
                        self.critical_errors.append(f"Cannot read {file}: {e}")
                else:
                    file_verification[category][file] = {"exists": False}
                    self.critical_errors.append(f"Missing required file: {file}")
        
        self.verification_results["data_integrity"] = file_verification
        
        # Print verification results
        print("\n📁 DATA INTEGRITY VERIFICATION")
        print("=" * 50)
        
        for category, files in file_verification.items():
            print(f"\n{category.upper()}:")
            for file, status in files.items():
                if status.get("exists") and status.get("readable"):
                    print(f"  ✅ {file}")
                    if "shape" in status:
                        print(f"     Shape: {status['shape']}, Columns: {status['columns']}")
                    elif "keys" in status:
                        print(f"     Keys: {status['keys']}")
                else:
                    print(f"  ❌ {file} - {status.get('error', 'Missing')}")
        
        return len(self.critical_errors) == 0
    
    def verify_phase3_readiness(self):
        """
        Verify readiness for Phase 3 implementation
        """
        logger.info("=== Verifying Phase 3 Readiness ===")
        
        readiness_checks = {}
        
        try:
            # Check aligned master dataset
            master_data = pd.read_csv(self.results_dir / "aligned_master_dataset_FIXED.csv", index_col=0, parse_dates=True)
            readiness_checks["master_dataset"] = {
                "shape": list(master_data.shape),
                "date_range": f"{master_data.index.min()} to {master_data.index.max()}",
                "has_regimes": "ECONOMIC_REGIME" in master_data.columns,
                "has_factors": all(col in master_data.columns for col in ["Value", "Quality", "MinVol", "Momentum"]),
                "has_market_data": "VIX" in master_data.columns and "SP500_Monthly_Return" in master_data.columns
            }
            
            # Check regime diversity in aligned data
            if readiness_checks["master_dataset"]["has_regimes"]:
                regime_counts = master_data["ECONOMIC_REGIME"].value_counts()
                readiness_checks["regime_diversity"] = {
                    "regime_distribution": regime_counts.to_dict(),
                    "all_four_regimes": len(regime_counts) == 4,
                    "sufficient_observations": all(count >= 10 for count in regime_counts.values())
                }
            
            # Check data completeness for visualizations
            completeness = {}
            for col in ["Value", "Quality", "MinVol", "Momentum", "SP500_Monthly_Return", "VIX"]:
                if col in master_data.columns:
                    completeness[col] = float((~master_data[col].isnull()).mean())
            
            readiness_checks["data_completeness"] = completeness
            readiness_checks["visualization_ready"] = all(complete > 0.8 for complete in completeness.values())
            
            self.verification_results["phase3_readiness"] = readiness_checks
            
            # Print results
            print("\n🚀 PHASE 3 READINESS VERIFICATION")
            print("=" * 50)
            
            print("MASTER DATASET:")
            print(f"  Shape: {readiness_checks['master_dataset']['shape']}")
            print(f"  Date Range: {readiness_checks['master_dataset']['date_range']}")
            print(f"  Has Regimes: {'✅' if readiness_checks['master_dataset']['has_regimes'] else '❌'}")
            print(f"  Has Factors: {'✅' if readiness_checks['master_dataset']['has_factors'] else '❌'}")
            print(f"  Has Market Data: {'✅' if readiness_checks['master_dataset']['has_market_data'] else '❌'}")
            
            if "regime_diversity" in readiness_checks:
                print("\nREGIME DIVERSITY:")
                for regime, count in readiness_checks["regime_diversity"]["regime_distribution"].items():
                    print(f"  {regime}: {count} observations")
                print(f"  All Four Regimes: {'✅' if readiness_checks['regime_diversity']['all_four_regimes'] else '❌'}")
                print(f"  Sufficient Data: {'✅' if readiness_checks['regime_diversity']['sufficient_observations'] else '❌'}")
            
            print("\nDATA COMPLETENESS:")
            for col, completeness in readiness_checks["data_completeness"].items():
                print(f"  {col}: {completeness:.1%}")
            
            print(f"\nVisualization Ready: {'✅' if readiness_checks['visualization_ready'] else '❌'}")
            
            return readiness_checks["visualization_ready"]
            
        except Exception as e:
            self.critical_errors.append(f"Error verifying Phase 3 readiness: {e}")
            return False
    
    def run_comprehensive_verification(self):
        """
        Run complete verification of Phase 2 and readiness for Phase 3
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE PHASE 2 VERIFICATION")
        logger.info("=" * 80)
        
        verification_results = {
            "data_integrity": self.verify_data_integrity(),
            "phase3_readiness": self.verify_phase3_readiness()
        }
        
        # Calculate overall verification status
        all_passed = all(verification_results.values())
        critical_issues = len(self.critical_errors)
        warnings_count = len(self.warnings)
        
        # Create comprehensive report
        comprehensive_report = {
            "verification_timestamp": datetime.now().isoformat(),
            "overall_status": "PASS" if all_passed else "FAIL",
            "critical_errors": self.critical_errors,
            "warnings": self.warnings,
            "verification_summary": verification_results,
            "detailed_results": self.verification_results
        }
        
        # Save verification report
        report_path = self.results_dir / 'comprehensive_phase2_verification.json'
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        # Final summary
        print("\n" + "=" * 80)
        print("🔍 COMPREHENSIVE VERIFICATION RESULTS")
        print("=" * 80)
        
        for check, passed in verification_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {check.replace('_', ' ').title()}")
        
        print(f"\n📊 SUMMARY:")
        print(f"Overall Status: {'✅ READY FOR PHASE 3' if all_passed else '❌ ISSUES FOUND'}")
        print(f"Critical Errors: {critical_issues}")
        print(f"Warnings: {warnings_count}")
        
        if self.critical_errors:
            print(f"\n🚨 CRITICAL ISSUES TO RESOLVE:")
            for i, error in enumerate(self.critical_errors, 1):
                print(f"  {i}. {error}")
        
        if all_passed:
            print(f"\n🎉 PHASE 2 VERIFICATION COMPLETE!")
            print(f"✅ All systems ready for Phase 3 implementation")
            print(f"🚀 Proceed with Phase 3 Alpha - Foundation & Core Heatmaps")
        else:
            print(f"\n⚠️  ISSUES FOUND - RESOLVE BEFORE PHASE 3")
            print(f"🔧 Fix critical errors before proceeding")
        
        print("=" * 80)
        
        logger.info("✅ Comprehensive verification completed")
        logger.info(f"📄 Report saved: {report_path}")
        
        return all_passed, comprehensive_report

def main():
    """
    Run comprehensive Phase 2 verification
    """
    verifier = ComprehensivePhase2Verifier()
    success, report = verifier.run_comprehensive_verification()
    
    if success:
        print("\n🎯 VERIFICATION SUCCESS!")
        print("📋 Phase 2 complete and verified")
        print("🚀 Ready to begin Phase 3 implementation")
    else:
        print("\n⚠️  VERIFICATION ISSUES FOUND!")
        print("🔧 Please resolve critical errors before Phase 3")
    
    return success

if __name__ == "__main__":
    main() 