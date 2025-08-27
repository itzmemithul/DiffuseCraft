import os
import torch
import gradio as gr
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
from huggingface_hub import HfApi
from openai import OpenAI
from PIL import Image
from io import BytesIO
import requests

from dotenv import load_dotenv
load_dotenv()

huggingfaceApKey = os.getenv("HUGGINGFACE_API_KEY")
hugging_face_user = os.getenv("HUGGING_FACE_USERNAME")
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

ikea_models = []
sd1point5_base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getHuggingfaceModels():    
    api = HfApi()
    models = api.list_models(author=hugging_face_user, use_auth_token=huggingfaceApKey)

    prefix = hugging_face_user + "/" + "ikea_room_designs_sd"

    for model in models:
        if model.modelId.startswith(prefix):
            model_name = model.modelId.replace(hugging_face_user + "/", "")
            ikea_models.append(model_name)
    ikea_models.append(sd1point5_base_model)        
    return ikea_models

def improve_prompt(prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=77,
            messages=[
                {"role": "system", "content": "You are a room interior designer"},
                {"role": "user", "content": f"Generate a description for a room based on the following input. Keep it under 3 sentences: {prompt}"}
            ]
        )
        generated_prompt = completion.choices[0].message.content
        print('AI Generated Prompt: ', generated_prompt)
        return generated_prompt
    except Exception as e:
        return f"Error generating AI prompt: {str(e)}"

def generate_ai_prompt(prompt, use_ai_prompt):
    if use_ai_prompt and prompt.strip() != "":
        prompt = improve_prompt(prompt)
        return prompt
    else:
        return ""




def generate_image(user_prompt, selected_model, cfg, num_inference_steps, input_image=None):
    # If AI prompt is used, prefer that over user prompt
    # if use_ai_prompt and user_prompt.strip() != "" and generated_prompt.strip() != "":
    #     prompt = generated_prompt
    # else:

    print(user_prompt)
    print(selected_model)
    print(cfg)
    print(num_inference_steps)
    print(input_image)


    prompt = user_prompt

    # If an image is provided, use img2img pipeline for image refinement
    if input_image is not None:
        # Resize the input image
        init_image = input_image.resize((768, 512))

        # Use img2img pipeline to refine the image based on the prompt
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        # pipe = StableDiffusionImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        pipe.enable_attention_slicing()
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        refined_image = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=cfg).images[0]
        output_path = "output/refined_image.png"
        refined_image.save(output_path)

        return refined_image, gr.update(selected=1)
        # return refined_image, prompt, gr.update(selected=1)

    # If no image is provided, use the normal text-to-image generation
    if selected_model.startswith(sd1point5_base_model):
        model = selected_model
    else:
        model = hugging_face_user + "/" + selected_model

    if "lora" in selected_model:
        pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        pipe.load_lora_weights(model)
    else:
        pipe = DiffusionPipeline.from_pretrained(model)

    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    image = pipe(prompt, num_inference_steps=num_inference_steps, cfg=cfg).images[0]
    output_path = "output/ai_generated_image.png"
    image.save(output_path)

    return image, gr.update(selected=1)
    # return image, prompt, gr.update(selected=1)





# Gradio interface
models = getHuggingfaceModels()
# logo_path = "Picture2.png"

logo_path = "logo.png"
background_image_path = "bg.png"
header_background_path = "header.png"  

custom_theme = gr.themes.Default().set(
    body_background_fill="#342c30",
    body_background_fill_dark="rgba(0,0,0,0)",
    body_text_color="black",
    body_text_color_dark="black",
    checkbox_label_text_color='black',
    checkbox_border_color_dark='black',
    block_background_fill="#D9D9D9",
    block_background_fill_dark="#D9D9D9",
    block_border_color='rgba(0,0,0,0)',
    block_border_color_dark='rgba(0,0,0,0)',
    border_color_primary_dark='rgba(0,0,0,0)',
    block_info_text_color='black',
    block_info_text_color_dark='black',
    block_label_text_color='black',
    block_label_text_color_dark='black',
    block_title_text_color='black',
    block_title_text_color_dark='black',
    border_color_primary='rgba(0,0,0,0)',
    block_label_background_fill="#898989",
    block_label_background_fill_dark="#898989",
    input_background_fill="#FFF",
    input_background_fill_dark="#FFF",
    input_border_color="rgba(0,0,0,0)",
    input_border_color_dark="rgba(0,0,0,0)",
    button_primary_background_fill="#312D2A",
    button_primary_background_fill_dark="#312D2A",
    slider_color_dark='#312D2A'
)


# Custom CSS and theme
# custom_css = """
#     #component-0 {
#         background-color: #003333 !important;
#     }04d
#     .logo {
#         position: fixed;
#         top: 10px;
#         right: 10px;
#         width: 50px;
#         height: auto;
#         z-index: 1000;
#     }
# """
with gr.Blocks(css=f"""
    .gradio-container {{
        background-image: url('file={background_image_path}');
        background-size: cover;
        background-color: #e6e4e1;
        padding: 0!important;
    }}
    .tabitem{{
        padding: 0!important;
    }}
    .header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-image: url('file={header_background_path}');
        background-size: cover;
        padding: 10px;
        font-size: 20px;
        font-weight: bold;
        color: white!important;
    }}
    .main-content{{
        padding:20px;
    }}
    #generate-button {{
        background: #312D2A !important;
        color: white !important;
        border: none;
        padding: 10px;
        cursor: pointer;
    }}
    #generate-button1 {{
        background: rgb(44, 71, 82) !important;
        color: white !important;
        border: none;
        padding: 10px;
        cursor: pointer;
    }}
    #generate-button1:hover{{
        background: #312D2A!important;
    }}
    .gr-input, .gr-slider, .gr-textbox {{
        background: rgba(0,0,0,0) !important;
        color: black !important;
        border: 1px solid #ccc;
    }}
    .selected{{
        background: #342c30!important;
        color: white!important;
    }}
    #component-13{{
        height: 175px!important;
    }}
    .gpu_selector{{
        display:flex;
        flex-direction: column;
        align-items:center;
    }}
""", theme=custom_theme) as interface:

    with gr.Tabs() as tabs:
        with gr.TabItem("Input", id=0):
            with gr.Row():
                    gr.Markdown(
                        f"<div class='header'>"
                        f"<img src='file={logo_path}' style='height: 30px;'>"
                        f" DiffuseCraft - Refurbish Your Space with AI! "
                        f"<span style='float: right; color: white;'>"
                        f"<em style='color: white;'>Namaste, <strong style='color: white;'>Mithul</strong></em>"
                        f"<img src='file=profile.png' style='height: 30px; display:inline;'>"
                        f"</span></div>"
                    )


            with gr.Row(elem_classes="main-content"):
                    with gr.Column(scale=1):
                        model_list = gr.Dropdown(models, value=ikea_models[0], label="Select Model", info="Choose the Image generation model you want to try!")
                        
                        cfg = gr.Slider(1, 20, value=7.5, label="Guidance Scale", info="Choose between 1 and 20")

                        num_inference_steps = gr.Slider(10, 100, value=20, label="Inference Steps", info="Choose between 10 and 100")

                        # use_ai_prompt = gr.Checkbox(label="Use AI to Generate Detailed Prompt", value=True)
                        
                        # generated_prompt = gr.Textbox(label="AI generated detailed prompt", placeholder="AI generated prompt will appear here...", interactive=False)
                    
                    with gr.Column(scale=1):
                        user_prompt = gr.Textbox(label="Enter your prompt", placeholder="Ex: Modern living room with sofa and coffee table ")

                        image_input = gr.Image(label="Upload Image (Optional)", type="pil", style={"height": "175px!important"})  # Added image input

            with gr.Row(elem_classes="main-content"):
                generate_button = gr.Button("Generate Image", elem_id="generate-button", loading=True, interactive=True)

        with gr.TabItem("Output", id=1):
            with gr.Row():
                    gr.Markdown(
                        f"<div class='header'>"
                        f"<img src='file={logo_path}' style='height: 30px;'>"
                        f" DiffuseCraft - Refurbish Your Space with AI! "
                        f"<span style='float: right; color: white;'>"
                        f"<em style='color: white;'>Namaste, <strong style='color: white;'>Mithul</strong></em>"
                        f"<img src='file=profile.png' style='height: 30px; display:inline;'>"
                        f"</span></div>"
                    )
            with gr.Row(elem_classes="main-content"):
                with gr.Column():
                    generated_output = gr.Image(label="Generated Output", width=450, height=450, type="pil") 
                with gr.Column():
                    refined_user_prompt = gr.Textbox(label="Refine your prompt", placeholder="Ex: Modern living room with sofa and coffee table ")

                    refine_prompt_button = gr.Button("Refine Image", elem_id="generate-button", loading=True, interactive=True)



        with gr.TabItem("Admin", id=2):
            with gr.Row():
                    gr.Markdown(
                        f"<div class='header'>"
                        f"<img src='file={logo_path}' style='height: 30px;'>"
                        f" DiffuseCraft - Refurbish Your Space with AI! "
                        f"<span style='float: right; color: white;'>"
                        f"<em style='color: white;'>Namaste, <strong style='color: white;'>Mithul</strong></em>"
                        f"<img src='file=profile.png' style='height: 30px; display:inline;'>"
                        f"</span></div>"
                    )
            with gr.Row(elem_classes="main-content"):
                with gr.Column():
                    with gr.Row():
                        admin_data_upload = gr.UploadButton("Select Dataset", elem_id="generate-button")
                        model_list = gr.Dropdown(models, value=ikea_models[0], label="Select Model", info="Choose the Image generation model you want to try!")
                    with gr.Row(elem_classes="gpu_selector"):
                        with gr.Row():
                            title = gr.Markdown(f"# Select your GPU")
                        with gr.Row():
                            with gr.Column():
                                t4 = gr.Button("T4", elem_id="generate-button1")
                            with gr.Column():
                                a10G = gr.Button("A10G", elem_id="generate-button1")
                            with gr.Column():
                                l4 = gr.Button("L4", elem_id="generate-button1")
                        with gr.Row():
                            with gr.Column():
                                l40s = gr.Button("L40S", elem_id="generate-button1")
                            with gr.Column():
                                a100 = gr.Button("A100", elem_id="generate-button1")
                            with gr.Column():
                                h100 = gr.Button("H100", elem_id="generate-button1")
                    with gr.Row():
                        submit_button_admin = gr.Button("Submit", elem_id="generate-button")


    # user_prompt.submit(fn=generate_ai_prompt, inputs=[user_prompt,], outputs=[])
    # user_prompt.submit(fn=generate_ai_prompt, inputs=[user_prompt, use_ai_prompt], outputs=[generated_prompt])
    # Button is initially enabled
    # generate_button.click(
    #     fn=disable_button,
    #     inputs=None,
    #     outputs=generate_button
    # )
    

    # generate_button.click(
    #     fn=generate_image,
    #     inputs=[user_prompt, model_list, cfg, num_inference_steps, image_input],
    #     outputs=[generated_output, tabs]
    # )

    generate_button.click(fn=lambda: gr.update(interactive=False), inputs=None, outputs=generate_button).then(
        fn=generate_image,
        inputs=[user_prompt, model_list, cfg, num_inference_steps, image_input],
        outputs=[generated_output, tabs]
    ).then(fn=lambda: gr.update(interactive=True), inputs=None, outputs=generate_button)

    refine_prompt_button.click(fn=lambda: gr.update(interactive=False), inputs=None, outputs=refine_prompt_button).then(
        fn=generate_image,
        inputs=[refined_user_prompt, model_list, cfg, num_inference_steps, generated_output],
        outputs=[generated_output, tabs]
    ).then(fn=lambda: gr.update(interactive=True), inputs=None, outputs=refine_prompt_button)

    # generate_button.click(
    #     fn=generate_image,
    #     inputs=[user_prompt, use_ai_prompt, generated_prompt, model_list, cfg, num_inference_steps, image_input],
    #     outputs=[generated_output, generated_prompt, tabs]
    # )

# with gr.Blocks(theme=custom_theme, css=custom_css) as demo:
#     gr.HTML(f"<img src='file={logo_path}' class='logo' alt='Logo'>")
#     gr.Markdown("# DiffuseCraft - Refurbish Your Space with AI!")
    
#     with gr.Row():
#         with gr.Column():
#             model_list = gr.Dropdown(models, value=ikea_models[0], label="Select Model", info="Choose the Image generation model you want to try!")
#         with gr.Column():
#             use_ai_prompt = gr.Checkbox(value=True, label="Use AI to generate detailed prompt", info="Check this box to generate a detailed prompt from AI based on your input")

#     with gr.Row():
#         with gr.Column():
#             user_prompt = gr.Textbox(label="Enter your prompt", placeholder="Enter a room description or theme...")
#             examples = gr.Examples(
#                 examples=["Modern living room with sofa and coffee table", "Cozy bedroom with ample of sunlight"],
#                 inputs=[user_prompt],
#             )
#         with gr.Column():
#             image_input = gr.Image(label="Upload Image (Optional)", type="pil")  # Added image input
#             generated_prompt = gr.Textbox(label="AI generated detailed prompt", placeholder="AI generated prompt will appear here...", interactive=False)
        
    # with gr.Row():
    #     with gr.Column():
    #         cfg = gr.Slider(1, 20, value=7.5, label="Guidance Scale", info="Choose between 1 and 20")
    #         num_inference_steps = gr.Slider(10, 100, value=20, label="Inference Steps", info="Choose between 10 and 100")

    #         generate_image_button = gr.Button(value="Generate Image")
    #     with gr.Column():    
    #         generated_output = gr.Image(label="Generated Image", width=512, height=512, type="pil")

#     # with gr.Row():
#     #     with gr.Column():
#     #         refine_image = gr.Button(value="Refine Image")
#     #     with gr.Column():
#     #         refine_image_output = gr.Image(label="Refined Image", width=512, height=512)

#     # Update submission logic to handle both image and prompt
#     user_prompt.submit(fn=generate_ai_prompt, inputs=[user_prompt, use_ai_prompt], outputs=[generated_prompt])
#     generate_image_button.click(fn=generate_image, inputs=[user_prompt, use_ai_prompt, generated_prompt, model_list, cfg, num_inference_steps, image_input], outputs=[generated_output, generated_prompt])
#     # refine_image.click(fn=generate_image, inputs=[user_prompt, use_ai_prompt, generated_prompt, model_list, cfg, num_inference_steps, image_input], outputs=[refine_image_output])


if __name__ == "__main__":
    interface.launch(share=True)
