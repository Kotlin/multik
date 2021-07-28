package org.jetbrains.kotlinx.multik.cuda

import java.util.*

internal class LinkedCache<T> : Iterable<T> {
    sealed class Tag<T> {
        abstract val parent: LinkedCache<T>?
    }

    val isEmpty
        get() = size == 0

    val first
        get() = head!!.value

    fun add(value: T): Tag<T> {
        return addNode(Node(value))
    }

    fun popFirst(): T {
        return removeNode(head!!).value
    }

    fun placeLast(tag: Tag<T>) {
        @Suppress("UNCHECKED_CAST")
        addNode(removeNode(tag as Node<T>))
    }

    fun remove(tag: Tag<T>) {
        @Suppress("UNCHECKED_CAST")
        removeNode(removeNode(tag as Node<T>))
    }

    override fun iterator(): Iterator<T> {
        return object : Iterator<T> {
            var current = head

            override fun hasNext(): Boolean = current != null

            override fun next(): T {
                val result = current!!.value
                current = current!!.next

                return result
            }
        }
    }

    override fun toString(): String {
        val builder = StringJoiner(", ", "[", "]")

        for (x in this)
            builder.add(x.toString())

        return builder.toString()
    }

    private class Node<T>(var value: T) : Tag<T>() {
        var next: Node<T>? = null
        var prev: Node<T>? = null

        override var parent: LinkedCache<T>? = null
    }

    private var head: Node<T>? = null
    private var tail: Node<T>? = null

    private var size: Int = 0

    private fun addNode(node: Node<T>): Node<T> {
        require(node.parent == null)

        if (size == 0) {
            head = node
        } else {
            tail!!.next = node
            node.prev = tail
            tail = node
        }

        tail = node
        node.parent = this

        size++

        return node
    }

    private fun removeNode(node: Node<T>): Node<T> {
        require(node.parent == this)

        if (node == head)
            head = node.next

        if (node == tail)
            tail = node.prev

        node.apply {
            prev?.next = next
            next?.prev = prev

            prev = null
            next = null
        }

        node.parent = null
        size -= 1

        return node
    }
}