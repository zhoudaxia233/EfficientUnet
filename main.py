from efficientunet import *

# This is a demo to show you how to use the library.
model = get_efficient_unet_b7((224, 224, 3), pretrained=True, block_type='transpose', concat_input=True)
model.summary()
