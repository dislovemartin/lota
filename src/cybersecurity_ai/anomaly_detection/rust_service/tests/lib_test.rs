// lib_test.rs (Rust Anomaly Detector Tests)
use crate::detect_anomaly;

#[test]
fn test_detect_anomaly() {
    let result = detect_anomaly("Test data");
    assert_eq!(result, vec!["Anomaly Detected".to_string()]);
}

