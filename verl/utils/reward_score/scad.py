
from .ioutils.normalize import *
from .ioutils.compute_IOU import *



def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """
    Compute score for SCAD solutions with proper temporary file management.
    
    Args:
        solution_str: The solution string containing SCAD code
        ground_truth: The ground truth SCAD code
        method: Scoring method (unused in this implementation)
        format_score: Score for format correctness
        score: Maximum score for correct solution
        
    Returns:
        float: Computed score between 0.0 and score
    """
    content = solution_str
    sol = ground_truth
    open_count, close_count, open_count_think, close_count_think = count_answer_tags(solution_str)

    content = extract_solution(content)
    
    if content is None:
        return 0.0
    
    # Initialize variables for resource tracking
    content_file_obj = None
    sol_file_obj = None
    content_file_path = None
    sol_file_path = None
    content_stl = None
    sol_stl = None
    
    try:
        # Create temporary files with proper error handling
        content_file_obj = tempfile.NamedTemporaryFile(mode='w', suffix='.scad', delete=False)
        content_file_path = content_file_obj.name
        content_file_obj.write(content)
        content_file_obj.close()  # Close the file so OpenSCAD can access it

        content_stl = normalize_openscad_to_stl(content_file_path)
        if not content_stl:
            return 0.0
        
        sol_file_obj = tempfile.NamedTemporaryFile(mode='w', suffix='.scad', delete=False)
        sol_file_path = sol_file_obj.name
        sol_file_obj.write(sol)
        sol_file_obj.close()  # Close the file so OpenSCAD can access it
            
        sol_stl = normalize_openscad_to_stl(sol_file_path)
        
        if not sol_stl:
            return 0.0
        
        # Calculate IOU score
        iou_value = calculate_3d_iou(sol_stl, content_stl)

        # Apply penalties for excessive answer tags
        if open_count > 5 or close_count > 5 or open_count_think > 5 or close_count_think > 5:
            iou_value = iou_value / 4
            
        return iou_value
        
    except Exception as e:
        # Log the error for debugging (consider using proper logging in production)
        import logging
        logging.warning(f"Error in SCAD score computation: {e}")
        return 0.0
    finally:
        # Ensure all temporary files are cleaned up properly
        temp_files_to_remove = [
            content_file_path, 
            sol_file_path, 
            content_stl, 
            sol_stl
        ]
        
        for temp_file in temp_files_to_remove:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except (OSError, PermissionError) as cleanup_error:
                    # Log cleanup errors but don't fail the function
                    import logging
                    logging.warning(f"Failed to remove temporary file {temp_file}: {cleanup_error}")
        
        # Ensure file objects are properly closed
        for file_obj in [content_file_obj, sol_file_obj]:
            if file_obj and not file_obj.closed:
                try:
                    file_obj.close()
                except:
                    pass


