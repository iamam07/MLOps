import torch
from transformers import AutoTokenizer
from model import CrossAttentionModel
import numpy as np

def test_model_inference():
    """Test the model with known sentence pairs and expected similarity ranges"""
    
    # Initialize model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossAttentionModel(model_name="sentence-transformers/all-MiniLM-L12-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    
    # Load trained model if exists
    try:
        checkpoint = torch.load("checkpoints/best_model_cross.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded trained model from checkpoint")
    except FileNotFoundError:
        print("⚠ No checkpoint found, using untrained model")
    
    model.to(device)
    model.eval()
    
    # Test cases: (sentence1, sentence2, expected_similarity_range)
    test_cases = [
        # High similarity (4-5 range)
        ("A dog is running in the park", "A canine is jogging in the garden", (3.5, 5.0)),
        ("The cat is sleeping", "The feline is resting", (3.5, 5.0)),
        
        # Medium similarity (2-4 range)
        ("I love pizza", "Pizza is delicious", (2.0, 4.5)),
        ("It's raining outside", "The weather is wet", (2.0, 4.5)),
        
        # Low similarity (0-2 range)
        ("I like programming", "The sky is blue", (0.0, 2.5)),
        ("Cars are fast", "Mathematics is difficult", (0.0, 2.5)),
        
        # Identical sentences (should be ~5)
        ("Hello world", "Hello world", (4.5, 5.0)),
        
        # Completely different (should be ~0)
        ("Elephant", "Computer programming", (0.0, 1.5))
    ]
    
    print("\n" + "="*60)
    print("INFERENCE TEST RESULTS")
    print("="*60)
    
    all_predictions = []
    passed_tests = 0
    
    with torch.no_grad():
        for i, (sent1, sent2, expected_range) in enumerate(test_cases, 1):
            # Tokenize sentences
            encoding1 = tokenizer(
                sent1, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            encoding2 = tokenizer(
                sent2, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            
            # Move to device
            input_ids1 = encoding1['input_ids'].to(device)
            attention_mask1 = encoding1['attention_mask'].to(device)
            input_ids2 = encoding2['input_ids'].to(device)
            attention_mask2 = encoding2['attention_mask'].to(device)
            
            # Get prediction
            prediction = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            pred_value = prediction.cpu().item()
            all_predictions.append(pred_value)
            
            # Check if prediction is in expected range
            in_range = expected_range[0] <= pred_value <= expected_range[1]
            if in_range:
                passed_tests += 1
                status = "✓ PASS"
            else:
                status = "✗ FAIL"
            
            print(f"\nTest {i}: {status}")
            print(f"  Sentence 1: '{sent1}'")
            print(f"  Sentence 2: '{sent2}'")
            print(f"  Predicted:  {pred_value:.3f}")
            print(f"  Expected:   {expected_range[0]:.1f} - {expected_range[1]:.1f}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed_tests}/{len(test_cases)} ({passed_tests/len(test_cases)*100:.1f}%)")
    print(f"Prediction range: {min(all_predictions):.3f} - {max(all_predictions):.3f}")
    print(f"Prediction mean: {np.mean(all_predictions):.3f}")
    print(f"Prediction std: {np.std(all_predictions):.3f}")
    
    # Check for common issues
    print("\n" + "="*60)
    print("DIAGNOSTIC CHECKS")
    print("="*60)
    
    # Check if all predictions are the same (dead model)
    if len(set([round(p, 3) for p in all_predictions])) == 1:
        print("⚠ WARNING: All predictions are identical - model may not be learning")
    else:
        print("✓ Model produces varied predictions")
    
    # Check if predictions are in valid range
    valid_range = all(0 <= p <= 5 for p in all_predictions)
    if valid_range:
        print("✓ All predictions are in valid range [0, 5]")
    else:
        print("⚠ WARNING: Some predictions are outside [0, 5] range")
    
    # Check if model shows reasonable behavior
    identical_score = all_predictions[6]  # "Hello world" vs "Hello world"
    different_score = all_predictions[7]  # "Elephant" vs "Computer programming"
    
    if identical_score > different_score:
        print("✓ Model correctly scores identical sentences higher than different ones")
    else:
        print("⚠ WARNING: Model doesn't distinguish between identical and different sentences")
    
    return all_predictions, passed_tests, len(test_cases)

def test_batch_inference():
    """Test batch processing capability"""
    print("\n" + "="*60)
    print("BATCH INFERENCE TEST")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossAttentionModel(model_name="sentence-transformers/all-MiniLM-L12-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    
    try:
        checkpoint = torch.load("checkpoints/best_model_cross.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        pass
    
    model.to(device)
    model.eval()
    
    # Test batch processing
    sentences1 = ["I love cats", "The weather is nice", "Programming is fun"]
    sentences2 = ["Cats are amazing", "It's a sunny day", "Coding is enjoyable"]
    
    # Batch tokenization
    encoding1 = tokenizer(sentences1, padding=True, truncation=True, max_length=128, return_tensors="pt")
    encoding2 = tokenizer(sentences2, padding=True, truncation=True, max_length=128, return_tensors="pt")
    
    # Move to device
    input_ids1 = encoding1['input_ids'].to(device)
    attention_mask1 = encoding1['attention_mask'].to(device)
    input_ids2 = encoding2['input_ids'].to(device)
    attention_mask2 = encoding2['attention_mask'].to(device)
    
    with torch.no_grad():
        predictions = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
    
    print("Batch predictions:")
    for i, (s1, s2, pred) in enumerate(zip(sentences1, sentences2, predictions)):
        print(f"  {i+1}. '{s1}' <-> '{s2}': {pred.item():.3f}")
    
    print("✓ Batch inference working correctly")

if __name__ == "__main__":
    print("Starting model inference tests...")
    
    try:
        # Run single inference tests
        predictions, passed, total = test_model_inference()
        
        # Run batch inference test
        test_batch_inference()
        
        print(f"\n{'='*60}")
        print("FINAL RESULT")
        print(f"{'='*60}")
        if passed >= total * 0.6:  # 60% pass rate threshold
            print("✓ MODEL APPEARS TO BE WORKING CORRECTLY")
        else:
            print("⚠ MODEL MAY NEED MORE TRAINING OR DEBUGGING")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()