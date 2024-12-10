#include <displayer/PointCloudScene.hpp>
#include <lux/engine/ui/Widget.hpp>
#include <lux/engine/ui/Window.hpp>
#include <lux/engine/ui/RenderContext.hpp>
#include <lux/engine/gapi/vk/ShaderLoader.hpp>
#include <lux/engine/math/eigen_extend.hpp>

#include <imgui_impl_vulkan.h>
#include <shader_config.hpp>
#include <stdexcept>
#include <random>

PointCloudScene::PointCloudScene(lux::ui::Window& window)
    : lux::ui::Widget("PCS(Point Cloud Scene)", window) {
    auto& render_context = window.renderContext();
    auto& fixed_context = render_context.fixedContext();

    createCommandPool();
    createRenderPass();
    createDepthResources();
    createUniformBuffer();
    createDescriptorSetLayout();
    createDescriptorSet();
    loadShaders();
    std::vector<VkDescriptorSetLayout> layouts{descriptor_set_layout_};
    createPipelineLayout(layouts);
    createPipeline();

    // 创建一个Sampler用于ImGui显示纹理
    {
        VkSamplerCreateInfo sampler_info{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        };
        if (vkCreateSampler(fixed_context.logical_device, &sampler_info, fixed_context.allocator, &sampler_) != VK_SUCCESS) {
            throw std::runtime_error("failed to create sampler!");
        }
    }

    createViewportResources();
}

PointCloudScene::~PointCloudScene()
{
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();
    vkDeviceWaitIdle(fixed_context.logical_device);

    // 销毁资源
    destroyViewportResources();
    if (sampler_) vkDestroySampler(fixed_context.logical_device, sampler_, fixed_context.allocator);

    if (descriptor_set_layout_) vkDestroyDescriptorSetLayout(fixed_context.logical_device, descriptor_set_layout_, fixed_context.allocator);

    pipeline_.release(fixed_context.logical_device);
    pipeline_layout_.release(fixed_context.logical_device);
    vertex_shader_.release(fixed_context.logical_device);
    fragment_shader_.release(fixed_context.logical_device);
    render_pass_.release(fixed_context.logical_device);
    command_pool_.release(fixed_context.logical_device);

    if (uniformBuffersMapped) {
        vkUnmapMemory(fixed_context.logical_device, uniform_buffer_memory_);
    }
    uniform_buffer_.release(fixed_context.logical_device, fixed_context.allocator);
    if (uniform_buffer_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(fixed_context.logical_device, uniform_buffer_memory_, fixed_context.allocator);
    }

    if (depth_image_view_) vkDestroyImageView(fixed_context.logical_device, depth_image_view_, fixed_context.allocator);
    if (depth_image_) {
        vkDestroyImage(fixed_context.logical_device, depth_image_, fixed_context.allocator);
    }
    if (depth_image_memory_) {
        vkFreeMemory(fixed_context.logical_device, depth_image_memory_, fixed_context.allocator);
    }
}

void PointCloudScene::createCommandPool()
{
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();

    command_pool_ = lux::gapi::vk::CommandPool::Builder()
        .setQueueFamilyIndex(fixed_context.graphics_queue_family_index)
        .build(fixed_context.logical_device, fixed_context.allocator);
}

void PointCloudScene::createRenderPass()
{
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();
    auto& gui_context = render_context.imguiContext();

    lux::gapi::vk::RenderPassBuilder builder;

    auto color_format = gui_context.window_data.SurfaceFormat.format;

    VkAttachmentDescription color_attachment{};
    color_attachment.format = color_format;
    color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // 离屏渲染后会作为纹理使用

    builder.addAttachment(color_attachment);

    VkAttachmentDescription depth_attachment{};
    depth_attachment.format = VK_FORMAT_D32_SFLOAT; // 一个深度格式即可
    depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    builder.addAttachment(depth_attachment);

    VkAttachmentReference color_reference{
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };

    VkAttachmentReference depth_reference{
        .attachment = 1,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_reference;
    subpass.pDepthStencilAttachment = &depth_reference;

    builder.addSubpass(subpass);

    render_pass_ = builder.build(fixed_context.logical_device);
}

void PointCloudScene::createDepthResources()
{
    // 创建与viewport大小匹配的深度图像，这里先不创建，等到真正有width_, height_时再创建
    // 在创建Viewport资源时进行
}

void PointCloudScene::createUniformBuffer()
{
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();
    VkDeviceSize uniformBufferSize = sizeof(Eigen::Matrix4f);

    uniform_buffer_ = lux::gapi::vk::BufferBuilder()
        .setSize(uniformBufferSize)
        .setUsage(lux::gapi::vk::EBufferUsage::UNIFORM_BUFFER)
        .setSharingMode(lux::gapi::vk::EBufferSharingMode::EXCLUSIVE)
        .build(fixed_context.logical_device, fixed_context.allocator);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(fixed_context.logical_device, uniform_buffer_.handle(), &memReq);

    uint32_t memoryTypeIndex = fixed_context.physical_device.findMemoryTypeIndex(
        memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    assert(memoryTypeIndex != UINT32_MAX);

    VkMemoryAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memReq.size,
        .memoryTypeIndex = memoryTypeIndex
    };

    if (vkAllocateMemory(fixed_context.logical_device, &allocInfo, fixed_context.allocator, &uniform_buffer_memory_) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate uniform buffer memory");
    }

    vkBindBufferMemory(fixed_context.logical_device, uniform_buffer_.handle(), uniform_buffer_memory_, 0);

    // 映射一下，后面更新MVP时可以直接memcpy
    vkMapMemory(fixed_context.logical_device, uniform_buffer_memory_, 0, uniformBufferSize, 0, &uniformBuffersMapped);
}

void PointCloudScene::createDescriptorSetLayout()
{
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();

    VkDescriptorSetLayoutBinding uboLayoutBinding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .pImmutableSamplers = nullptr
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &uboLayoutBinding,
    };

    if (vkCreateDescriptorSetLayout(fixed_context.logical_device, &layoutInfo, fixed_context.allocator, &descriptor_set_layout_) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void PointCloudScene::createDescriptorSet()
{
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();

    VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = fixed_context.descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_set_layout_
    };

    if (vkAllocateDescriptorSets(fixed_context.logical_device, &allocInfo, &descriptor_set_) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor set!");
    }

    VkDescriptorBufferInfo bufferInfo{
        .buffer = uniform_buffer_.handle(),
        .offset = 0,
        .range = sizeof(Eigen::Matrix4f)
    };

    VkWriteDescriptorSet descriptorWrite{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set_,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &bufferInfo,
    };

    vkUpdateDescriptorSets(fixed_context.logical_device, 1, &descriptorWrite, 0, nullptr);
}

void PointCloudScene::loadShaders()
{
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();

    vertex_shader_   = lux::gapi::vk::createShaderModule(fixed_context.logical_device, vertex_shader_path);
    fragment_shader_ = lux::gapi::vk::createShaderModule(fixed_context.logical_device, fragment_shader_path);
}

void PointCloudScene::createPipelineLayout(const std::vector<VkDescriptorSetLayout>& layouts)
{
    auto& render_context = _window.renderContext();
    auto& fixed_context  = render_context.fixedContext();

    // 有一个descriptor set layout用于UBO
    lux::gapi::vk::PipelineLayoutBuilder layoutBuilder;
    layoutBuilder.setLayouts(layouts);

    pipeline_layout_ = layoutBuilder.build(fixed_context.logical_device);
}

void PointCloudScene::createPipeline()
{
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertex_shader_,
        .pName = "main"
    };

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragment_shader_,
        .pName = "main"
    };

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    auto& binding_description = PointVertex::getBindingDescription();
    auto& attribute_descriptions = PointVertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding_description,
        .vertexAttributeDescriptionCount = (uint32_t)attribute_descriptions.size(),
        .pVertexAttributeDescriptions = attribute_descriptions.data(),
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        .primitiveRestartEnable = VK_FALSE
    };

    // 动态指定viewport和scissor
    VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    VkPipelineRasterizationStateCreateInfo rasterizer{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f
    };

    VkPipelineMultisampleStateCreateInfo multisampling{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT
    };

    VkPipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
    };

    VkPipelineColorBlendStateCreateInfo colorBlending{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment
    };

    VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        .stencilTestEnable = VK_FALSE
    };

    std::array<VkDynamicState, 2> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = (uint32_t)dynamicStates.size(),
        .pDynamicStates = dynamicStates.data()
    };

    pipeline_ = lux::gapi::vk::GraphicsPipeline::Builder()
        .addShaderStage(shaderStages[0])
        .addShaderStage(shaderStages[1])
        .setVertexInputState(vertexInputInfo)
        .setInputAssemblyState(inputAssembly)
        .setViewportState(viewportState)
        .setRasterizationState(rasterizer)
        .setMultisampleState(multisampling)
        .setDepthStencilState(depthStencil)
        .setColorBlendState(colorBlending)
        .setDynamicState(dynamicState)
        .setLayout(pipeline_layout_)
        .build(fixed_context.logical_device, render_pass_.handle(), fixed_context.allocator);
}

void PointCloudScene::createViewportResources()
{
    // 首次创建时需要width_, height_ > 0
    if (width_ == 0 || height_ == 0) return;

    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();
    auto& gui_context = render_context.imguiContext();

    auto image_count = 1; // 离屏渲染只需要一张图像
    auto color_format = gui_context.window_data.SurfaceFormat.format;

    viewport_images_.resize(image_count);
    dst_image_memory_.resize(image_count);
    viewport_image_views_.resize(image_count);

    // 创建color图像
    {
        VkImageCreateInfo imageCreateCI{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = color_format,
            .extent = { (uint32_t)width_, (uint32_t)height_, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        };

        if (vkCreateImage(fixed_context.logical_device, &imageCreateCI, fixed_context.allocator, &viewport_images_[0]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create offscreen color image!");
        }

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(fixed_context.logical_device, viewport_images_[0], &memReq);
        uint32_t memTypeIndex = fixed_context.physical_device.findMemoryTypeIndex(
            memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );

        VkMemoryAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memReq.size,
            .memoryTypeIndex = memTypeIndex
        };

        if (vkAllocateMemory(fixed_context.logical_device, &allocInfo, fixed_context.allocator, &dst_image_memory_[0]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate memory for offscreen color image!");
        }

        vkBindImageMemory(fixed_context.logical_device, viewport_images_[0], dst_image_memory_[0], 0);

        // 创建ImageView
        VkImageViewCreateInfo view_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = viewport_images_[0],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = color_format,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .levelCount = 1,
                .layerCount = 1,
            }
        };

        if (vkCreateImageView(fixed_context.logical_device, &view_info, fixed_context.allocator, &viewport_image_views_[0]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create offscreen image view!");
        }

        tex_id_ = ImGui_ImplVulkan_AddTexture(
            sampler_,
            viewport_image_views_[0],
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );
    }

    // 创建深度图像
    {
        VkImageCreateInfo depthImageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_D32_SFLOAT,
            .extent = { (uint32_t)width_, (uint32_t)height_, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        };

        if (vkCreateImage(fixed_context.logical_device, &depthImageInfo, fixed_context.allocator, &depth_image_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create depth image!");
        }

        VkMemoryRequirements depthMemReq;
        vkGetImageMemoryRequirements(fixed_context.logical_device, depth_image_, &depthMemReq);
        uint32_t depthMemType = fixed_context.physical_device.findMemoryTypeIndex(
            depthMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        VkMemoryAllocateInfo depthAllocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = depthMemReq.size,
            .memoryTypeIndex = depthMemType
        };

        if (vkAllocateMemory(fixed_context.logical_device, &depthAllocInfo, fixed_context.allocator, &depth_image_memory_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate depth image memory!");
        }
        vkBindImageMemory(fixed_context.logical_device, depth_image_, depth_image_memory_, 0);

        VkImageViewCreateInfo depthViewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = depth_image_,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_D32_SFLOAT,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .levelCount = 1,
                .layerCount = 1
            }
        };

        if (vkCreateImageView(fixed_context.logical_device, &depthViewInfo, fixed_context.allocator, &depth_image_view_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create depth image view!");
        }
    }

    // 创建FrameBuffer
    {
        std::array<VkImageView, 2> attachments = {
            viewport_image_views_[0],
            depth_image_view_
        };

        frame_buffer_ = lux::gapi::vk::FrameBuffer::Builder()
            .setRenderPass(render_pass_.handle())
            .setAttachments(attachments.data(), (uint32_t)attachments.size())
            .setSize(width_, height_)
            .setLayers(1)
            .build(fixed_context.logical_device, fixed_context.allocator);
    }

    // 创建DescriptorSet用于ImGui显示该纹理
    // ImGui_ImplVulkan_AddTexture创建的描述符是COMBINED_IMAGE_SAMPLER
    // 这里用AddTexture方便ImGui显示
}

void PointCloudScene::recreateViewportResources()
{
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();
    vkDeviceWaitIdle(fixed_context.logical_device);

    destroyViewportResources();
    if (width_ > 0 && height_ > 0) {
        createViewportResources();
    }
}

static std::vector<PointVertex> generate_random_cloud_points(size_t number)
{
    std::vector<PointVertex> points;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < number; ++i) {
        PointVertex point;
        point.position = { dist(gen) * 2.0f - 1.0f, dist(gen) * 2.0f - 1.0f, -(dist(gen) * 5.0f + 0.1f) };
        point.color = { dist(gen), dist(gen), dist(gen) };
        points.push_back(point);
    }

    return points;
}

void PointCloudScene::destroyViewportResources()
{
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();
    if (frame_buffer_.handle() != VK_NULL_HANDLE) {
        frame_buffer_.release(fixed_context.logical_device, fixed_context.allocator);
    }

    for (auto& iv : viewport_image_views_) {
        if (iv != VK_NULL_HANDLE) {
            vkDestroyImageView(fixed_context.logical_device, iv, fixed_context.allocator);
        }
    }
    viewport_image_views_.clear();

    for (size_t i = 0; i < viewport_images_.size(); i++) {
        if (viewport_images_[i] != VK_NULL_HANDLE) {
            vkDestroyImage(fixed_context.logical_device, viewport_images_[i], fixed_context.allocator);
        }
        if (dst_image_memory_[i] != VK_NULL_HANDLE) {
            vkFreeMemory(fixed_context.logical_device, dst_image_memory_[i], fixed_context.allocator);
        }
    }
    viewport_images_.clear();
    dst_image_memory_.clear();

    if (depth_image_view_) {
        vkDestroyImageView(fixed_context.logical_device, depth_image_view_, fixed_context.allocator);
        depth_image_view_ = VK_NULL_HANDLE;
    }
    if (depth_image_) {
        vkDestroyImage(fixed_context.logical_device, depth_image_, fixed_context.allocator);
        depth_image_ = VK_NULL_HANDLE;
    }
    if (depth_image_memory_) {
        vkFreeMemory(fixed_context.logical_device, depth_image_memory_, fixed_context.allocator);
        depth_image_memory_ = VK_NULL_HANDLE;
    }
}

void PointCloudScene::bindMVPMatrixToShader()
{
    // 更新UBO中的MVP矩阵
    // 使用Eigen构建MVP
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    Eigen::Vector3f eye(0, 0, -3);
    Eigen::Vector3f center(0, 0, 0);
    Eigen::Vector3f up(0, 1, 0);
    Eigen::Affine3f view = LuxEigenExt::lookAtf(eye, center, up);

    Eigen::Matrix4f proj = LuxEigenExt::perspectiveProjectionf(
        60.0f * __PI_FLOAT_CONSTANT / 180.0f,
        width_ > 0 && height_ > 0 ? (float)width_ / (float)height_ : 1.0f,
        0.1f,
        100.0f
    );
    proj(1, 1) *= -1; // Vulkan Y翻转

    Eigen::Matrix4f mvp = proj * view * model;

    if (uniformBuffersMapped) {
        memcpy(uniformBuffersMapped, mvp.data(), sizeof(mvp));
    }
}

void PointCloudScene::createVertexBuffer(const std::vector<PointVertex>& points)
{
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();

    VkDeviceSize bufferSize = sizeof(PointVertex) * points.size();

    // 创建顶点缓冲区
    vertex_buffer_ = lux::gapi::vk::BufferBuilder()
        .setSize(bufferSize)
        .setUsage(lux::gapi::vk::EBufferUsage::VERTEX_BUFFER)
        .setSharingMode(lux::gapi::vk::EBufferSharingMode::EXCLUSIVE)
        .build(fixed_context.logical_device, fixed_context.allocator);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(fixed_context.logical_device, vertex_buffer_.handle(), &memReq);

    uint32_t memoryTypeIndex = fixed_context.physical_device.findMemoryTypeIndex(
        memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    assert(memoryTypeIndex != UINT32_MAX);

    VkMemoryAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memReq.size,
        .memoryTypeIndex = memoryTypeIndex
    };

    if (vkAllocateMemory(fixed_context.logical_device, &allocInfo, fixed_context.allocator, &vertex_buffer_memory_) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate vertex buffer memory");
    }

    vkBindBufferMemory(fixed_context.logical_device, vertex_buffer_.handle(), vertex_buffer_memory_, 0);

    // 拷贝点云数据到缓冲区
    vkMapMemory(fixed_context.logical_device, vertex_buffer_memory_, 0, bufferSize, 0, &vertexBufferMapped);
    memcpy(vertexBufferMapped, points.data(), (size_t)bufferSize);
    vkUnmapMemory(fixed_context.logical_device, vertex_buffer_memory_);
}

void PointCloudScene::destroyVertexBuffer() {
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();

    if (vertex_buffer_.handle() != VK_NULL_HANDLE) {
        vertex_buffer_.release(fixed_context.logical_device, fixed_context.allocator);
    }
    if (vertex_buffer_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(fixed_context.logical_device, vertex_buffer_memory_, fixed_context.allocator);
        vertex_buffer_memory_ = VK_NULL_HANDLE;
    }
}

void PointCloudScene::issueDrawCommands()
{
    auto& render_context = _window.renderContext();
    auto& fixed_context = render_context.fixedContext();

    // 创建并录制命令缓冲
    VkCommandBufferAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool_,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(fixed_context.logical_device, &alloc_info, &cmd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer!");
    }

    VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };

    vkBeginCommandBuffer(cmd, &begin_info);

    // 动态设置viewport与scissor
    VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = (float)width_,
        .height = (float)height_,
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };

    VkRect2D scissor{
        .offset = {0,0},
        .extent = {(uint32_t)width_, (uint32_t)height_}
    };

    VkRenderPassBeginInfo render_pass_info{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = render_pass_.handle(),
        .framebuffer = frame_buffer_.handle(),
        .renderArea = scissor
    };

    VkClearValue clearValues[2];
    clearValues[0].color = { {0.0f,0.0f,0.0f,1.0f} };
    clearValues[1].depthStencil = { 1.0f, 0 };
    render_pass_info.clearValueCount = 2;
    render_pass_info.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_.handle());

    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // 绑定描述符集 (ubo)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_.handle(), 0, 1, &descriptor_set_, 0, nullptr);

    // 没有实际的点云数据,假设有N个点:
    uint32_t vertexCount = 1000; // 示例
    auto points = generate_random_cloud_points(vertexCount);
    createVertexBuffer(points);
    VkDeviceSize offsets[] = { 0 };
    // 如果有vertex_buffer_，需要提前创建并绑定内存和填充数据
    vkCmdBindVertexBuffers(cmd,0,1,&vertex_buffer_, offsets);

    vkCmdDraw(cmd, vertexCount, 1, 0, 0);
    vkCmdEndRenderPass(cmd);

    vkEndCommandBuffer(cmd);

    // 提交命令并等待完成(简化处理)
    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd
    };

    vkQueueSubmit(fixed_context.graphics_queue.handle(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(fixed_context.graphics_queue.handle());
    vkFreeCommandBuffers(fixed_context.logical_device, command_pool_, 1, &cmd);

    destroyVertexBuffer();
}

void PointCloudScene::renderPointCloud()
{
    bindMVPMatrixToShader();
    issueDrawCommands();
}

void PointCloudScene::paint()
{
    ImVec2 size = ImGui::GetWindowSize();
    width_ = (int)size.x;
    height_ = (int)size.y;

    if (width_ != prev_width_ || height_ != prev_height_) {
        recreateViewportResources();
        prev_width_ = width_;
        prev_height_ = height_;
    }

    if (width_ == 0 || height_ == 0) {
        ImGui::Text("Window too small.");
        return;
    }

    // 离屏渲染
    renderPointCloud();

    ImVec2 image_size((float)width_, (float)height_);
    ImGui::Image((ImTextureID)tex_id_, image_size);
}
