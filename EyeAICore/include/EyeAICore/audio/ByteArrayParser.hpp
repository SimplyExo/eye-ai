#pragma once

#include <sndfile.hh>
#include <cstddef>
#include <cstring>

/*
Handles the conversion of the bytearray, representing
the .wav file where the audio labels are stored, to
a state, where it can be processed by the libsndfile
library
*/

struct MemoryData {
    const std::byte* data;
    sf_count_t size;
    sf_count_t pos;
};

inline sf_count_t vio_get_filelen(void* user_data) {
    auto* mem = static_cast<MemoryData*>(user_data);
    return mem->size;
}

inline sf_count_t vio_seek(sf_count_t offset, int whence, void* user_data) {
    auto* mem = static_cast<MemoryData*>(user_data);
    sf_count_t newpos = mem->pos;

    switch (whence) {
        case SEEK_SET: newpos = offset; break;
        case SEEK_CUR: newpos += offset; break;
        case SEEK_END: newpos = mem->size + offset; break;
        default: return -1;
    }

    if (newpos < 0 || newpos > mem->size)
        return -1;

    mem->pos = newpos;
    return mem->pos;
}

inline sf_count_t vio_read(void* ptr, sf_count_t count, void* user_data) {
    auto* mem = static_cast<MemoryData*>(user_data);
    sf_count_t remain = mem->size - mem->pos;
    sf_count_t to_read = (count < remain ? count : remain);

    if (to_read > 0) {
        std::memcpy(ptr, mem->data + mem->pos, static_cast<size_t>(to_read));
        mem->pos += to_read;
    }

    return to_read;
}

inline sf_count_t vio_write(const void*, sf_count_t, void*) {
    return 0;
}

inline sf_count_t vio_tell(void* user_data) {
    auto* mem = static_cast<MemoryData*>(user_data);
    return mem->pos;
}
