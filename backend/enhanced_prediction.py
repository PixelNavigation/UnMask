import numpy as np
import torch
from dfdc_model.kernel_utils import isotropically_resize_image, put_to_center, normalize_transform
from gradcam import GradCAM, create_gradcam_image, image_to_base64

def enhanced_predict_on_video(face_extractor, video_path, batch_size, input_size, models, strategy=np.mean,
                              apply_compression=False, return_frame_predictions=True):
    """Enhanced video prediction that returns frame-level predictions and Grad-CAM"""
    # Reduce batch size for lower memory usage
    batch_size = max(8, batch_size * 2)
    frame_predictions = []
    fake_frames = []
    gradcam_images = []
    
    try:
        faces = face_extractor.process_video(video_path)
        # Sample every Nth frame to reduce memory usage
        frame_sample_rate = 3  # Only process every 3rd frame
        if len(faces) > 0:
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)
            frame_data_list = []
            n = 0
            for idx, frame_data in enumerate(faces):
                if idx % frame_sample_rate != 0:
                    continue
                for face in frame_data["faces"]:
                    resized_face = isotropically_resize_image(face, input_size)
                    resized_face = put_to_center(resized_face, input_size)
                    if apply_compression:
                        from albumentations.augmentations.functional import image_compression
                        resized_face = image_compression(resized_face, quality=90, image_type=".jpg")
                    if n + 1 < batch_size:
                        x[n] = resized_face
                        frame_data_list.append({
                            'frame_idx': frame_data["frame_idx"],
                            'original_face': face,
                            'processed_face': resized_face
                        })
                        n += 1
                    else:
                        break
            if n > 0:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                x = torch.tensor(x, device=device).float()
                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))
                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)
                
                # Make predictions for each frame
                all_preds = []
                for model in models:
                    if device == "cuda":
                        y_pred = model(x[:n].half())
                    else:
                        y_pred = model(x[:n])
                    y_pred = torch.sigmoid(y_pred.squeeze())
                    frame_preds = y_pred.detach().cpu().numpy()
                    
                    # Handle single frame case
                    if n == 1:
                        frame_preds = [frame_preds]
                    
                    all_preds.append(frame_preds)
                
                # Average predictions across models
                avg_frame_preds = np.mean(all_preds, axis=0)
                
                # Collect all frame predictions for strategy calculation
                all_predictions = []
                
                # Identify fake frames (threshold > 0.5)
                for i, pred in enumerate(avg_frame_preds):
                    if i < len(frame_data_list):
                        frame_idx = frame_data_list[i]['frame_idx']
                        frame_predictions.append({
                            'frame_idx': frame_idx,
                            'prediction': float(pred)
                        })
                        all_predictions.append(float(pred))
                    
                        if pred > 0.5:  # Fake frame
                            fake_frames.append(frame_idx)
                
                # Generate Grad-CAM for the most suspicious frame
                if len(fake_frames) > 0 and len(models) > 0:
                    # Find the frame with highest fake probability
                    max_pred_idx = np.argmax(avg_frame_preds)
                    if max_pred_idx < len(frame_data_list):
                        try:
                            print(f"Generating Grad-CAM for frame {max_pred_idx} with prediction {avg_frame_preds[max_pred_idx]}")
                            # Get the target layer (last convolutional layer)
                            model = models[0]
                            target_layer = None
                            for name, module in model.named_modules():
                                if isinstance(module, torch.nn.Conv2d):
                                    target_layer = module
                        
                            if target_layer is not None:
                                print(f"Using target layer: {target_layer}")
                                gradcam = GradCAM(model, target_layer)
                                input_tensor = x[max_pred_idx:max_pred_idx+1].clone().detach()
                                input_tensor.requires_grad = True
                                if device == "cuda":
                                    input_tensor = input_tensor.half()
                                heatmap = gradcam.generate_cam(input_tensor)
                                # Create visualization
                                original_face = frame_data_list[max_pred_idx]['processed_face']
                                # Ensure heatmap is detached before converting to numpy
                                if isinstance(heatmap, torch.Tensor):
                                    heatmap = heatmap.detach().cpu().numpy()
                                gradcam_image = create_gradcam_image(original_face, heatmap)
                                gradcam_base64 = image_to_base64(gradcam_image)
                                original_base64 = image_to_base64(original_face)
                                print(f"Generated Grad-CAM: original={len(original_base64) if original_base64 else 0}, gradcam={len(gradcam_base64) if gradcam_base64 else 0}")
                                gradcam_images.append({
                                    'original': original_base64,
                                    'gradcam': gradcam_base64
                                })
                            else:
                                print("No convolutional layer found for Grad-CAM")
                        except Exception as e:
                            print(f"Grad-CAM generation error: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Return overall prediction using strategy
                overall_prediction = strategy(all_predictions)
                
                print(f"Video analysis complete: {len(fake_frames)}/{len(all_predictions)} suspicious frames, overall: {overall_prediction:.3f}")
                
                return {
                    'prediction': overall_prediction,
                    'frame_predictions': frame_predictions,
                    'fake_frames': fake_frames,
                    'gradcam_images': gradcam_images
                }
                    
    except Exception as e:
        print("Enhanced prediction error on video %s: %s" % (video_path, str(e)))

    return {
        'prediction': 0.5,
        'frame_predictions': [],
        'fake_frames': [],
        'gradcam_images': []
    }

def enhanced_predict_on_image(image_path, models, input_size=380):
    """Enhanced image prediction with Grad-CAM"""
    try:
        from dfdc_model.kernel_utils import isotropically_resize_image, put_to_center, normalize_transform
        from facenet_pytorch.models.mtcnn import MTCNN
        import cv2
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device=device)
        
        # Read and process image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(image)
        img_pil = img_pil.resize(size=[s // 2 for s in img_pil.size])
        
        batch_boxes, probs = detector.detect(img_pil, landmarks=False)
        
        if batch_boxes is None:
            return {
                'prediction': 0.5,
                'gradcam_image': None,
                'original_image': None
            }
            
        # Process the best face
        best_idx = 0
        if len(probs) > 1:
            best_idx = np.argmax(probs)
            
        bbox = batch_boxes[best_idx]
        if bbox is not None:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3
            crop = image[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            
            # Resize and center
            resized_face = isotropically_resize_image(crop, input_size)
            resized_face = put_to_center(resized_face, input_size)
            
            # Prepare tensor
            x = torch.tensor(resized_face, device=device).float().unsqueeze(0)
            x = x.permute((0, 3, 1, 2))
            x = normalize_transform(x[0] / 255.).unsqueeze(0)
            
            # Predict
            preds = []
            for model in models:
                if device == "cuda":
                    y_pred = model(x.half())
                else:
                    y_pred = model(x)
                y_pred = torch.sigmoid(y_pred.squeeze())
                preds.append(y_pred.detach().cpu().numpy())
            
            prediction = np.mean(preds)
            
            # Generate Grad-CAM for all predictions (not just fake ones)
            gradcam_image = None
            original_image = None
            if len(models) > 0:  # Generate for all predictions, not just fake ones
                try:
                    print(f"Generating Grad-CAM for image with prediction {prediction}")
                    model = models[0]
                    target_layer = None
                    for name, module in model.named_modules():
                        if isinstance(module, torch.nn.Conv2d):
                            target_layer = module
                    
                    if target_layer is not None:
                        print(f"Using target layer: {target_layer}")
                        gradcam = GradCAM(model, target_layer)
                        input_tensor = x.clone().detach()
                        input_tensor.requires_grad = True
                        if device == "cuda":
                            input_tensor = input_tensor.half()
                        heatmap = gradcam.generate_cam(input_tensor)
                        gradcam_vis = create_gradcam_image(resized_face, heatmap)
                        gradcam_image = image_to_base64(gradcam_vis)
                        original_image = image_to_base64(resized_face)
                        print(f"Generated Grad-CAM for image: original={len(original_image) if original_image else 0}, gradcam={len(gradcam_image) if gradcam_image else 0}")
                    else:
                        print("No convolutional layer found for Grad-CAM")
                except Exception as e:
                    print(f"Grad-CAM generation error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Skipping Grad-CAM: models={len(models)}")
            
            return {
                'prediction': prediction,
                'gradcam_image': gradcam_image,
                'original_image': original_image
            }
        
        return {
            'prediction': 0.5,
            'gradcam_image': None,
            'original_image': None
        }
        
    except Exception as e:
        print(f"Enhanced image prediction error: {e}")
        return {
            'prediction': 0.5,
            'gradcam_image': None,
            'original_image': None
        }
