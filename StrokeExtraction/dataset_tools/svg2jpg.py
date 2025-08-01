import os
import pandas as pd
from PIL import Image, ImageDraw
import svgpathtools
from xml.dom import minidom
import numpy as np

def process_all_kanji(input_dir='kanji', output_dir='output'):
    """
    Process all SVG files in the kanji folder with proper scaling
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'jp_strokes'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'jp_char'), exist_ok=True)
    
    # Initialize DataFrame
    all_strokes_df = pd.DataFrame(columns=['character', 'unicode', 'stroke', 'stroke_nums', 'target'])
    
    for svg_file in os.listdir(input_dir):
        if not svg_file.endswith('.svg'):
            continue
            
        try:
            svg_path = os.path.join(input_dir, svg_file)
            print(f"Processing {svg_file}...")
            
            stroke_df = process_single_kanji(
                svg_path=svg_path,
                output_dir=output_dir
            )
            
            all_strokes_df = pd.concat([all_strokes_df, stroke_df], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing {svg_file}: {str(e)}")
            continue
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'kanji_strokes.csv')
    all_strokes_df.to_csv(csv_path, index=False)
    print(f"Saved stroke data to {csv_path}")

def process_single_kanji(svg_path, output_dir):
    """
    Process a single kanji with proper scaling to 256x256
    """
    # Parse SVG and get dimensions
    doc = minidom.parse(svg_path)
    paths = doc.getElementsByTagName('path')
    filename = os.path.splitext(os.path.basename(svg_path))[0]
    
    # Get original dimensions from SVG
    svg_width, svg_height = get_svg_dimensions(doc)
    if svg_width == 0 or svg_height == 0:
        svg_width, svg_height = 200, 200  # Default if not found
    
    # Calculate scaling factors
    scale_x = 256 / svg_width
    scale_y = 256 / svg_height
    scale = min(scale_x, scale_y) * 0.9  # 90% to add some margin
    
    # Calculate offset to center the character
    offset_x = (256 - svg_width * scale) / 2
    offset_y = (256 - svg_height * scale) / 2
    
    # Create complete character image
    char_img = Image.new('L', (256, 256), 'white')
    char_draw = ImageDraw.Draw(char_img)
    
    # Prepare stroke data
    stroke_data = []
    total_strokes = len(paths)
    
    for stroke_num, path in enumerate(paths, start=1):
        stroke_img = Image.new('L', (256, 256), 'white')
        stroke_draw = ImageDraw.Draw(stroke_img)
        
        path_data = path.getAttribute('d')
        parsed_path = svgpathtools.parse_path(path_data)
        
        for segment in parsed_path:
            # Scale and translate coordinates
            start = (
                offset_x + segment.start.real * scale,
                offset_y + segment.start.imag * scale
            )
            end = (
                offset_x + segment.end.real * scale,
                offset_y + segment.end.imag * scale
            )
            
            stroke_draw.line([start, end], fill='black', width=3)
            char_draw.line([start, end], fill='black', width=3)
        
        # Save stroke image
        stroke_filename = f"{filename}_{stroke_num}.png"
        stroke_img.save(os.path.join(output_dir, 'jp_strokes', stroke_filename))
        
        stroke_data.append({
            'character': filename,
            'unicode': filename,
            'stroke': stroke_num,
            'stroke_nums': total_strokes,
            'target': f"{filename}_{stroke_num}"
        })
    
    # Save complete character image
    char_img.save(os.path.join(output_dir, 'jp_char', f"{filename}.jpg"))
    
    doc.unlink()
    return pd.DataFrame(stroke_data)

def get_svg_dimensions(doc):
    """
    Extract width and height from SVG document
    """
    svg = doc.getElementsByTagName('svg')[0]
    width = float(svg.getAttribute('width').replace('px', '')) if svg.getAttribute('width') else 0
    height = float(svg.getAttribute('height').replace('px', '')) if svg.getAttribute('height') else 0
    
    if width == 0 or height == 0:
        viewbox = svg.getAttribute('viewBox')
        if viewbox:
            _, _, vb_width, vb_height = map(float, viewbox.split())
            return vb_width, vb_height
    
    return width, height

if __name__ == "__main__":
    process_all_kanji()