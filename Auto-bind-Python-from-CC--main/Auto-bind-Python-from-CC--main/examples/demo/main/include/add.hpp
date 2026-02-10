// 头文件保护宏，防止头文件被重复包含
#ifndef ADD_H
#define ADD_H

#pragma once

namespace add::test
{
    /**
     * My function, add two integer.
     * @param a arg a, int type
     * @param b arg b, int type
     * @return int type, will a + b
     * @module add.test.add
     */
    int add(int a, int b);

}

#endif // ADD_H