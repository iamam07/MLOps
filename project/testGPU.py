import torch
from transformers import AutoTokenizer
from model import CrossEncoderRegressionModel
import numpy as np

def test_model_inference():
    """Test the model with known sentence pairs and expected similarity ranges"""
    
    # Initialize model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    model = CrossEncoderRegressionModel(model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load trained model if exists
    try:
        checkpoint = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print("✓ Loaded trained model from checkpoint")
    except FileNotFoundError:
        print("⚠ No checkpoint found, using untrained model")
    
    model.to(device)
    model.eval()
    
    # Test cases: (sentence1, sentence2, expected_similarity_range)
    test_cases = [
        ("A dog is running in the park", "A canine is jogging in the garden", (3.5, 5.0)),
        ("The cat is sleeping", "The feline is resting", (3.5, 5.0)),
        ("I love pizza", "Pizza is delicious", (2.0, 4.5)),
        ("It's raining outside", "The weather is wet", (2.0, 4.5)),
        ("I like programming", "The sky is blue", (0.0, 2.5)),
        ("Cars are fast", "Mathematics is difficult", (0.0, 2.5)),
        ("Hello world", "Hello world", (4.5, 5.0)),
        ("Elephant", "Computer programming", (0.0, 1.5))
    ]
    
    print("\n" + "="*60)
    print("INFERENCE TEST RESULTS")
    print("="*60)
    
    all_predictions = []
    passed_tests = 0
    
    with torch.no_grad():
        for i, (sent1, sent2, expected_range) in enumerate(test_cases, 1):
            # Tokenizar las dos oraciones juntas
            encoding = tokenizer(
                sent1, 
                sent2,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Mover a device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Obtener predicción
            prediction = model(input_ids, attention_mask)
            pred_value = prediction.cpu().item() * 5.0  # Escalar a [0, 5]
            all_predictions.append(pred_value)
            
            # Verificar si la predicción está en el rango esperado
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
    
    # Estadísticas resumidas
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed_tests}/{len(test_cases)} ({passed_tests/len(test_cases)*100:.1f}%)")
    print(f"Prediction range: {min(all_predictions):.3f} - {max(all_predictions):.3f}")
    print(f"Prediction mean: {np.mean(all_predictions):.3f}")
    print(f"Prediction std: {np.std(all_predictions):.3f}")
    
    # Verificaciones de diagnóstico
    print("\n" + "="*60)
    print("DIAGNOSTIC CHECKS")
    print("="*60)
    
    if len(set([round(p, 3) for p in all_predictions])) == 1:
        print("⚠ WARNING: All predictions are identical - model may not be learning")
    else:
        print("✓ Model produces varied predictions")
    
    valid_range = all(0 <= p <= 5 for p in all_predictions)
    if valid_range:
        print("✓ All predictions are in valid range [0, 5]")
    else:
        print("⚠ WARNING: Some predictions are outside [0, 5] range")
    
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
    model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    model = CrossEncoderRegressionModel(model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        checkpoint = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print("✓ Loaded trained model from checkpoint")
    except FileNotFoundError:
        print("⚠ No checkpoint found, using untrained model")
    
    model.to(device)
    model.eval()
    
    # Test batch processing
    sentences1 = ["I love cats", "The weather is nice", "Programming is fun"]
    sentences2 = ["Cats are amazing", "It's a sunny day", "Coding is enjoyable"]
    
    # Tokenización por lotes
    encoding = tokenizer(
        sentences1,
        sentences2,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Mover a device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        predictions = model(input_ids, attention_mask)
        predictions = predictions.cpu().numpy() * 5.0  # Escalar a [0, 5]
    
    print("Batch predictions:")
    for i, (s1, s2, pred) in enumerate(zip(sentences1, sentences2, predictions)):
        print(f"  {i+1}. '{s1}' <-> '{s2}': {pred.item():.3f}")
    
    print("✓ Batch inference working correctly")

if __name__ == "__main__":
    print("Starting model inference tests...")
    
    try:
        # Ejecutar pruebas de inferencia individual
        predictions, passed, total = test_model_inference()
        
        # Ejecutar prueba de inferencia por lotes
        test_batch_inference()
        
        print(f"\n{'='*60}")
        print("FINAL RESULT")
        print(f"{'='*60}")
        if passed >= total * 0.6:  # Umbral de aprobación del 60%
            print("✓ MODEL APPEARS TO BE WORKING CORRECTLY")
        else:
            print("⚠ MODEL MAY NEED MORE TRAINING OR DEBUGGING")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()