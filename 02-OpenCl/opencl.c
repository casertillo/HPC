/* utilities for opencl to get device, etc */

#include <stdio.h>
#include <string.h>

#include "lbm.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif




void get_opencl_platforms(cl_platform_id ** platforms, cl_uint * num_platforms)
{
    cl_int err;

    err = clGetPlatformIDs(0, NULL, num_platforms);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting number of platforms", err);

    *platforms = (cl_platform_id *) calloc(*num_platforms, sizeof(cl_platform_id));
    err = clGetPlatformIDs(*num_platforms, *platforms, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting platforms", err);
}

char * get_platform_info(cl_platform_info param_name, cl_platform_id platform)
{
    cl_int err;
    size_t return_size;

    err = clGetPlatformInfo(platform, param_name, 0, NULL, &return_size);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting return size of platform parameter", err);

    char * return_string = (char *) calloc(return_size, sizeof(char));
    err = clGetPlatformInfo(platform, param_name, return_size, return_string, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting platform info", err);

    return return_string;
}

void get_platform_devices(cl_platform_id platform, cl_device_id ** devices, cl_uint * num_devices)
{
    cl_int err;

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, num_devices);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting number of devices", err);

    *devices = (cl_device_id *) calloc(*num_devices, sizeof(cl_device_id));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, *num_devices, *devices, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting device IDs", err);
}

char * get_device_info(cl_device_info param_name, cl_device_id device)
{
    cl_int err;

    char * return_string = NULL;
    size_t return_size;

    err = clGetDeviceInfo(device, param_name, 0, NULL, &return_size);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting return size of device parameter", err);

    switch (param_name)
    {
    case CL_DEVICE_TYPE:
        return_string = (char *) calloc(50, sizeof(char));
        cl_device_type device_type;
        err = clGetDeviceInfo(device, param_name, return_size, &device_type, NULL);

        switch (device_type)
        {
        case CL_DEVICE_TYPE_GPU:
            strcat(return_string, "GPU");
            break;
        case CL_DEVICE_TYPE_CPU:
            strcat(return_string, "CPU");
            break;
        case CL_DEVICE_TYPE_ACCELERATOR:
            strcat(return_string, "ACCELERATOR");
            break;
        default:
            strcat(return_string, "DEFAULT");
            break;
        }
        break;
    case CL_DEVICE_NAME:
        return_string = (char *) calloc(return_size, sizeof(char));
        err = clGetDeviceInfo(device, param_name, return_size, return_string, NULL);
        break;
    default:
        DIE("Other device_info types not implemented\n");
    }

    if (CL_SUCCESS != err) DIE("OpenCL error %d getting device parameter", err);

    return return_string;
}

void print_device_info(cl_device_id device, int device_id)
{
    char * device_name = get_device_info(CL_DEVICE_NAME, device);
    char * device_type = get_device_info(CL_DEVICE_TYPE, device);

    fprintf(stdout, " Device %u: %s (%s)\n", device_id, device_name, device_type);

    free(device_name);
    free(device_type);
}

void list_opencl_platforms(void)
{
    cl_platform_id * platforms = NULL;
    cl_uint num_platforms;
    get_opencl_platforms(&platforms, &num_platforms);

    cl_uint i, d;

    for (i = 0; i < num_platforms; i++)
    {
        cl_uint num_devices = 0;
        cl_device_id * devices = NULL;

        get_platform_devices(platforms[i], &devices, &num_devices);

        char * profile = get_platform_info(CL_PLATFORM_PROFILE, platforms[i]);
        char * version = get_platform_info(CL_PLATFORM_VERSION, platforms[i]);
        char * name = get_platform_info(CL_PLATFORM_NAME, platforms[i]);
        char * vendor = get_platform_info(CL_PLATFORM_VENDOR, platforms[i]);

        fprintf(stdout, "Platform %u: %s - %s (OpenCL profile = %s, version = %s)\n",
            i, vendor, name, profile, version);

        for (d = 0; d < num_devices; d++)
        {
            print_device_info(devices[d], d);
        }

        free(profile);
        free(version);
        free(name);
        free(vendor);

        free(devices);
    }

    free(platforms);

    exit(EXIT_SUCCESS);
}

void opencl_initialise(int device_id, param_t params, accel_area_t accel_area,
    lbm_context_t * lbm_context, speed_t * cells, int * obstacles)
{
    /* get device etc. */
    cl_platform_id * platforms = NULL;
    cl_uint num_platforms;
    get_opencl_platforms(&platforms, &num_platforms);

    cl_device_id * devices = NULL;
    cl_uint total_devices;
    cl_uint i;


    total_devices = 0;

    for (i = 0; i < num_platforms; i++)
    {
        cl_uint num_platform_devices = 0;
        cl_device_id * platform_devices = NULL;

        get_platform_devices(platforms[i], &platform_devices, &num_platform_devices);

        devices = (cl_device_id *) realloc(devices, sizeof(cl_device_id)*(total_devices + num_platform_devices));
        memcpy(&devices[total_devices], platform_devices, num_platform_devices*sizeof(cl_device_id));

        total_devices += num_platform_devices;

        free(platform_devices);
    }

    if (device_id >= (int) total_devices)
    {
        DIE("Asked for device %d but there were only %u available!\n", device_id, total_devices);
    }

    lbm_context->device = devices[device_id];

    free(devices);
    free(platforms);

    fprintf(stdout, "Got OpenCL device:\n");
    print_device_info(lbm_context->device, device_id);

    cl_int err;

    /* create the context and command queue */
    lbm_context->context = clCreateContext(NULL, 1, &lbm_context->device, NULL, NULL, &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating context", err);

    lbm_context->queue = clCreateCommandQueue(lbm_context->context, lbm_context->device, 0, &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating command queue", err);

    fprintf(stdout, "Created OpenCL context\n");

    /* Get kernels from file etc */
    FILE * source_fp;

    #define KERNEL_FILE "kernels.cl"

    source_fp = fopen(KERNEL_FILE, "r");

    if (NULL == source_fp)
    {
        DIE("Unable to open kernel file %s", KERNEL_FILE);
    }

    size_t source_size;
    fseek(source_fp, 0, SEEK_END);
    source_size = ftell(source_fp);

    char * source = (char *) calloc(source_size + 1, sizeof(char));
    fseek(source_fp, 0, SEEK_SET);
    size_t bytes_read = fread(source, 1, source_size, source_fp);

    if (bytes_read != source_size)
    {
        DIE("Expected to read %lu bytes from kernel file, actually read %lu bytes", source_size, bytes_read);
    }

    source[source_size] = '\0';

    cl_program program = clCreateProgramWithSource(lbm_context->context, 1, (const char**)&source, NULL, &err);

    free(source);
    fclose(source_fp);

    if (CL_SUCCESS != err) DIE("OpenCL error %d creating program", err);

    fprintf(stdout, "Building program\n");

    err = clBuildProgram(program, 1, &lbm_context->device, NULL, NULL, NULL);

    if (err == CL_BUILD_PROGRAM_FAILURE)
    {
        cl_int build_err;
        size_t log_size;

        build_err = clGetProgramBuildInfo(program, lbm_context->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        if (CL_SUCCESS != build_err) DIE("OpenCL error %d getting size of build log", build_err);

        char * build_log = (char *) calloc(log_size + 1, sizeof(char));
        build_err = clGetProgramBuildInfo(program, lbm_context->device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        if (CL_SUCCESS != build_err) DIE("OpenCL error %d getting build log", build_err);

        printf("OpenCL program build log:\n%s\n", build_log);
        free(build_log);
    }

    if (CL_SUCCESS != err) DIE("OpenCL error %d building program", err);

    fprintf(stdout, "Finished initialising OpenCL\n");

        lbm_context->propagate = clCreateKernel(program, "propagate", &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating kernel", err);

        lbm_context->collision = clCreateKernel(program, "collision", &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating kernel", err);
    /*
    *   TODO
    *   Allocate memory and create kernels
    */
    // Create the compute kernel from the program 
    //WORK GROUP SIZE AND STEPS
    int nsteps;
    
    nsteps = params.nx*params.ny;

    printf("nsteps:%d\n", nsteps);
    
    //-------------------------------------------------------------
    //-------------------------------------------------------------
    int ii;
    speed_t *h_temp_cells;
        h_temp_cells = (speed_t *) malloc(sizeof(speed_t)*(params.ny*params.nx));
    if (h_temp_cells == NULL) DIE("Cannot allocate memory for temp_cells");
    /* Initialise arrays */
    for (ii =(params.nx*params.ny); ii --;)
    {
            /* centre */
            h_temp_cells[ii].speeds[0] = 0.0;
            /* axis directions */
            h_temp_cells[ii].speeds[1] = 0.0;
            h_temp_cells[ii].speeds[2] = 0.0;
            h_temp_cells[ii].speeds[3] = 0.0;
            h_temp_cells[ii].speeds[4] = 0.0;
            /* diagonals */
            h_temp_cells[ii].speeds[5] = 0.0;
            h_temp_cells[ii].speeds[6] = 0.0;
            h_temp_cells[ii].speeds[7] = 0.0;
            h_temp_cells[ii].speeds[8] = 0.0;
    }

    double *h_av_vels;
        
        h_av_vels = (double*)malloc(sizeof(double)*(params.max_iters*params.num_groups));
    if (h_av_vels == NULL) DIE("Cannot allocate memory for h_av_vels");

    for(ii=(params.max_iters*params.num_groups);ii--;)
    {
        h_av_vels[ii] = 0.0;
    }

    int *h_timestep;
        h_timestep = (int*)malloc(sizeof(int));
    if (h_timestep == NULL) DIE("Cannot allocate memory for h_timestep");
    //cluster memory
    h_timestep = 0;
    lbm_context->d_cells = clCreateBuffer(lbm_context->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                            sizeof(speed_t)*(params.nx*params.ny), cells, &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting buffer d_cells", err);

    lbm_context->d_temp_cells = clCreateBuffer(lbm_context->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                            sizeof(speed_t)*(params.nx*params.ny), h_temp_cells, &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting buffer d_temp_cells", err);
    
    lbm_context->d_obstacles = clCreateBuffer(lbm_context->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                            sizeof(int)*(params.nx*params.ny), obstacles, &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting buffer d_obstacles", err);

    lbm_context->d_av_vels = clCreateBuffer(lbm_context->context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,

                            sizeof(double)*(params.max_iters*params.num_groups), h_av_vels, &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting buffer d_av_vels", err);


    err  = clSetKernelArg(lbm_context->propagate, 0, sizeof(param_t), &params);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting params", err);
    err  = clSetKernelArg(lbm_context->propagate, 1, sizeof(accel_area_t), &accel_area);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting accel_area", err);
    err  = clSetKernelArg(lbm_context->propagate, 2, sizeof(cl_mem), &lbm_context->d_cells);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting d_cells", err);
    err  = clSetKernelArg(lbm_context->propagate, 3, sizeof(cl_mem), &lbm_context->d_temp_cells);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting d_temp_cells", err);
    err  = clSetKernelArg(lbm_context->propagate, 4, sizeof(cl_mem), &lbm_context->d_obstacles);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting d_obstalces", err); 

    err  = clSetKernelArg(lbm_context->collision, 0, sizeof(param_t), &params);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting params", err);
    err  = clSetKernelArg(lbm_context->collision, 1, sizeof(cl_mem), &lbm_context->d_cells);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting d_cells", err);
    err  = clSetKernelArg(lbm_context->collision, 2, sizeof(cl_mem), &lbm_context->d_temp_cells);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting d_temp_cells", err);
    err  = clSetKernelArg(lbm_context->collision, 3, sizeof(cl_mem), &lbm_context->d_obstacles);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting d_obstalces", err); 
    err  = clSetKernelArg(lbm_context->collision, 4, sizeof(cl_mem), &lbm_context->d_av_vels);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting d_av_vels", err); 

    //HARD CODE---CHANGE TO NUMBER OPTIMAL LOCAL SIZES
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    err  = clSetKernelArg(lbm_context->collision, 5, sizeof(double) * 32*32, NULL);
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting d_av_vels", err); 

        err = clFinish(lbm_context->queue);
        if (CL_SUCCESS != err) DIE("OpenCL error %d getting buffer clFinish", err);
}

void opencl_finalise(lbm_context_t lbm_context)
{
    clReleaseCommandQueue(lbm_context.queue);
    clReleaseContext(lbm_context.context);
}

