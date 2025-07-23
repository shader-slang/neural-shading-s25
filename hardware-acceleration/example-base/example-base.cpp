#include "example-base.h"

#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace Slang;
using namespace rhi;

int64_t getCurrentTime()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

int64_t getTimerFrequency()
{
    return std::chrono::high_resolution_clock::period::den;
}

class DebugCallback : public IDebugCallback
{
public:
    virtual SLANG_NO_THROW void SLANG_MCALL
    handleMessage(DebugMessageType type, DebugMessageSource source, const char* message) override
    {
        const char* typeStr = "";
        switch (type)
        {
        case DebugMessageType::Info:
            typeStr = "INFO: ";
            break;
        case DebugMessageType::Warning:
            typeStr = "WARNING: ";
            break;
        case DebugMessageType::Error:
            typeStr = "ERROR: ";
            break;
        default:
            break;
        }
        const char* sourceStr = "[GraphicsLayer]: ";
        switch (source)
        {
        case DebugMessageSource::Slang:
            sourceStr = "[Slang]: ";
            break;
        case DebugMessageSource::Driver:
            sourceStr = "[Driver]: ";
            break;
        }
        printf("%s%s%s\n", sourceStr, typeStr, message);
#ifdef _WIN32
        OutputDebugStringA(sourceStr);
        OutputDebugStringA(typeStr);
        OutputDebugStringW(String(message).toWString());
        OutputDebugStringW(L"\n");
#endif
    }
};

#ifdef _WIN32
void _Win32OutputDebugString(const char* str)
{
    OutputDebugStringW(Slang::String(str).toWString().begin());
}
#endif
